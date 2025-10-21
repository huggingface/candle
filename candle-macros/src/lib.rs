use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, parse::Parser};

/// Derive macro to automatically implement QuantizedType trait
/// 
/// Usage:
/// ```ignore
/// #[derive(QuantizedType)]
/// #[quantized(name = "q4_0")]
/// pub struct Q4_0;
/// ```
#[proc_macro_derive(QuantizedType, attributes(quantized))]
pub fn derive_quantized_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    
    let name = &input.ident;
    
    // Extract the name from attributes
    let type_name = input.attrs.iter()
        .find(|attr| attr.path().is_ident("quantized"))
        .and_then(|attr| {
            attr.parse_args::<syn::LitStr>().ok()
        })
        .map(|lit| lit.value())
        .unwrap_or_else(|| name.to_string().to_lowercase());
    
    let expanded = quote! {
        impl candle_core::dtype::QuantizedType for #name {
            const NAME: &'static str = #type_name;
        }
    };
    
    TokenStream::from(expanded)
}

/// Helper macro to generate backend-agnostic method wrappers
/// 
/// This generates methods that automatically call the right backend implementation
/// based on the storage type.
/// 
/// Usage:
/// ```ignore
/// impl Q4_0 {
///     quantized_method!(dequantize(data: &[u8], output: &mut [f32]) -> ());
///     quantized_method!(matmul(lhs: &[u8], lhs_shape: &[usize], rhs: &[u8], rhs_shape: &[usize]) -> Vec<u8>);
/// }
/// ```
#[proc_macro]
pub fn quantized_method(_input: TokenStream) -> TokenStream {
    // This is a placeholder - the real implementation would parse method signatures
    // For now, we'll generate everything in register_quantized_types
    TokenStream::new()
}

/// Macro to register multiple quantized types and generate optimal dispatch
/// 
/// This macro generates:
/// 1. The QuantizedDType enum with all registered types
/// 2. CPU dispatch functions (always available)
/// 3. CUDA dispatch functions (feature-gated)
/// 4. Metal dispatch functions (feature-gated)
/// 5. Backend integration in Map2/Map1Any traits
/// 
/// Usage:
/// ```ignore
/// register_quantized_types! {
///     Q4_0,
///     Q4_1,
///     Q8_0,
/// }
/// ```
#[proc_macro]
pub fn register_quantized_types(input: TokenStream) -> TokenStream {
    let types = syn::punctuated::Punctuated::<syn::Ident, syn::Token![,]>::parse_terminated
        .parse(input)
        .expect("Expected comma-separated list of type names");
    
    let type_names: Vec<_> = types.iter().collect();
    let type_count = type_names.len();
    
    // Generate enum variants
    let variants = type_names.iter().map(|name| {
        quote! { #name }
    });
    
    // Generate match arms for dequantize
    let dequantize_arms = type_names.iter().map(|name| {
        quote! {
            QuantizedDType::#name => #name::dequantize(data, output)
        }
    });
    
    // Generate match arms for quantize
    let quantize_arms = type_names.iter().map(|name| {
        quote! {
            QuantizedDType::#name => #name::quantize(input)
        }
    });
    
    // Generate match arms for storage_size
    let storage_size_arms = type_names.iter().map(|name| {
        quote! {
            QuantizedDType::#name => #name::storage_size_in_bytes(num_elements)
        }
    });
    
    // Generate match arms for get_name
    let name_arms = type_names.iter().map(|name| {
        quote! {
            QuantizedDType::#name => #name::NAME
        }
    });
    
    // Generate match arms for matmul (f32 × quantized → f32 only)
    let matmul_arms = type_names.iter().map(|name| {
        quote! {
            QuantizedDType::#name => #name::matmul(lhs_f32, lhs_shape, rhs_data, rhs_shape)
        }
    });
    
    // Generate const IDs
    let const_ids = type_names.iter().map(|name| {
        let const_name = syn::Ident::new(
            &format!("{}_ID", name.to_string().to_uppercase()),
            name.span()
        );
        quote! {
            pub const #const_name: QuantizedDType = QuantizedDType::#name;
        }
    });
    
    let expanded = quote! {
        /// Quantized data type - fully compile-time!
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        #[repr(u8)]
        pub enum QuantizedDType {
            #(#variants,)*
        }
        
        impl QuantizedDType {
            /// Total number of registered types
            pub const COUNT: usize = #type_count;
            
            /// Get the name of this quantized type (O(1) via match)
            #[inline]
            pub const fn name(self) -> &'static str {
                match self {
                    #(#name_arms,)*
                }
            }
        }
        
        // Export const IDs for convenience
        #(#const_ids)*
        
        /// Dispatch functions - all compile-time match statements (O(1) via jump table)
        /// 
        /// These functions automatically route to the correct backend implementation
        /// based on the data types (CPU slices, CudaSlice, Metal buffers, etc.)
        pub mod quantized_dispatch {
            use super::*;
            use crate::Result;
            
            // ==================== CPU Backend (Always Available) ====================
            
            #[inline]
            pub fn dequantize_cpu(
                id: QuantizedDType,
                data: &[u8],
                output: &mut [f32]
            ) -> Result<()> {
                match id {
                    #(#dequantize_arms,)*
                }
            }
            
            #[inline]
            pub fn quantize_cpu(
                id: QuantizedDType,
                input: &[f32]
            ) -> Result<Vec<u8>> {
                match id {
                    #(#quantize_arms,)*
                }
            }
            
            #[inline]
            pub fn storage_size_in_bytes(
                id: QuantizedDType,
                num_elements: usize
            ) -> usize {
                match id {
                    #(#storage_size_arms,)*
                }
            }
            
            /// Matrix multiplication: f32 × quantized → f32 (ONLY supported mixed precision)
            /// This is the common pattern for inference: f32_activations @ quantized_weights
            /// All other combinations (quantized × f32, quantized × quantized) use auto-dequantization
            #[inline]
            pub fn matmul_cpu(
                lhs_f32: &[f32],
                lhs_shape: &[usize],
                rhs_id: QuantizedDType,
                rhs_data: &[u8],
                rhs_shape: &[usize],
            ) -> Result<Vec<f32>> {
                match rhs_id {
                    #(#matmul_arms,)*
                }
            }
            
            // ==================== CUDA Backend (Feature-Gated) ====================
            
            #[cfg(feature = "cuda")]
            #[inline]
            pub fn dequantize_cuda(
                id: QuantizedDType,
                data: &cudarc::driver::CudaSlice<u8>,
                output: &mut cudarc::driver::CudaSlice<f32>
            ) -> Result<()> {
                match id {
                    #(
                        QuantizedDType::#type_names => {
                            // Try to call the CUDA-specific method if it exists
                            // If not, fall back to CPU dequantization
                            #type_names::dequantize_cuda(data, output)
                                .or_else(|_| {
                                    // Fallback: copy to CPU, dequantize, copy back
                                    let cpu_data = data.to_host()?;
                                    let mut cpu_output = vec![0.0f32; output.len()];
                                    #type_names::dequantize_cpu(&cpu_data, &mut cpu_output)?;
                                    output.copy_from_host(&cpu_output)?;
                                    Ok(())
                                })
                        }
                    )*
                }
            }
            
            #[cfg(feature = "cuda")]
            #[inline]
            pub fn matmul_cuda(
                id: QuantizedDType,
                lhs_data: &cudarc::driver::CudaSlice<u8>,
                lhs_shape: &[usize],
                rhs_data: &cudarc::driver::CudaSlice<u8>,
                rhs_shape: &[usize],
            ) -> Result<cudarc::driver::CudaSlice<u8>> {
                match id {
                    #(
                        QuantizedDType::#type_names => {
                            #type_names::matmul_cuda(lhs_data, lhs_shape, rhs_data, rhs_shape)
                        }
                    )*
                }
            }
            
            // ==================== Metal Backend (Feature-Gated) ====================
            
            #[cfg(feature = "metal")]
            #[inline]
            pub fn dequantize_metal(
                id: QuantizedDType,
                data: &metal::Buffer,
                output: &mut metal::Buffer
            ) -> Result<()> {
                match id {
                    #(
                        QuantizedDType::#type_names => {
                            #type_names::dequantize_metal(data, output)
                        }
                    )*
                }
            }
            
            #[cfg(feature = "metal")]
            #[inline]
            pub fn matmul_metal(
                id: QuantizedDType,
                lhs_data: &metal::Buffer,
                lhs_shape: &[usize],
                rhs_data: &metal::Buffer,
                rhs_shape: &[usize],
            ) -> Result<metal::Buffer> {
                match id {
                    #(
                        QuantizedDType::#type_names => {
                            #type_names::matmul_metal(lhs_data, lhs_shape, rhs_data, rhs_shape)
                        }
                    )*
                }
            }
        }
        
        // Helper function to get name
        #[inline]
        pub fn get_quantized_name(id: QuantizedDType) -> &'static str {
            id.name()
        }
        
        // ==================== Backend Integration ====================
        
        /// Helper trait to check if a type has a CUDA implementation
        #[cfg(feature = "cuda")]
        pub trait HasCudaImpl {
            fn has_cuda_dequantize() -> bool { false }
            fn has_cuda_matmul() -> bool { false }
        }
        
        /// Helper trait to check if a type has a Metal implementation
        #[cfg(feature = "metal")]
        pub trait HasMetalImpl {
            fn has_metal_dequantize() -> bool { false }
            fn has_metal_matmul() -> bool { false }
        }
    };
    
    TokenStream::from(expanded)
}
