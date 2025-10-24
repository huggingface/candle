use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, parse::Parser};

/// Derive macro for QuantizedType trait implementation
/// 
/// Attributes:
/// - `name`: Type name identifier (default: lowercase struct name)
/// - `size_in_bytes`: Static size per element in bytes (default: 1)
/// 
/// Example: `#[derive(QuantizedType)] #[quantized(name = "q4_0", size_in_bytes = 1)] pub struct Q4_0;`
#[proc_macro_derive(QuantizedType, attributes(quantized))]
pub fn derive_quantized_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    
    let name = &input.ident;
    
    // Extract attributes from #[quantized(name = "...", size_in_bytes = N)]
    let mut type_name = name.to_string().to_lowercase();
    let mut size_in_bytes = 1usize; // Default to 1 byte
    
    for attr in &input.attrs {
        if attr.path().is_ident("quantized") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("name") {
                    let value = meta.value()?;
                    let s: syn::LitStr = value.parse()?;
                    type_name = s.value();
                } else if meta.path.is_ident("size_in_bytes") {
                    let value = meta.value()?;
                    let n: syn::LitInt = value.parse()?;
                    size_in_bytes = n.base10_parse()?;
                }
                Ok(())
            });
        }
    }
    
    let expanded = quote! {
        impl crate::dtype::QuantizedType for #name {
            const NAME: &'static str = #type_name;
            const SIZE_IN_BYTES: usize = #size_in_bytes;
        }
    };
    
    TokenStream::from(expanded)
}

/// Register multiple quantized types and generate dispatch
/// 
/// Generates:
/// - QuantizedDType enum with all types + external slots
/// - CPU/CUDA/Metal dispatch functions (feature-gated)
/// - External type support via HashMap registry
/// 
/// Example: `register_quantized_types! { Q4_0, Q4_1, Q8_0 }`
#[proc_macro]
pub fn register_quantized_types(input: TokenStream) -> TokenStream {
    let types = syn::punctuated::Punctuated::<syn::Ident, syn::Token![,]>::parse_terminated
        .parse(input)
        .expect("Expected comma-separated list of type names");
    
    let type_names: Vec<_> = types.iter().collect();
    let type_count = type_names.len();
    
    // Generate enum variants for built-in types
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
    
    let expanded = quote! {
        /// Quantized data type enum
        /// 
        /// Built-in types use compile-time dispatch.
        /// External types registered dynamically by name.
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
        #[repr(u8)]
        pub enum QuantizedDType {
            // Built-in types
            #(#variants,)*
            
            // External type (runtime registration)
            External(&'static str),
        }
        
        impl QuantizedDType {
            /// Built-in type count
            pub const BUILTIN_COUNT: usize = #type_count;
            
            /// Get type name
            #[inline]
            pub fn name(self) -> &'static str {
                match self {
                    #(#name_arms,)*
                    QuantizedDType::External(name) => name,
                }
            }
            
            /// Get size in bytes per element
            #[inline]
            pub fn size_in_bytes(self) -> usize {
                match self {
                    #(QuantizedDType::#type_names => #type_names::SIZE_IN_BYTES,)*
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get();
                        if let Some(registry) = registry {
                            let map = registry.read().unwrap();
                            if let Some(ops) = map.get(name) {
                                return ops.size_in_bytes;
                            }
                        }
                        1
                    }
                }
            }
            
            /// Check if external type
            #[inline]
            pub const fn is_external(self) -> bool {
                matches!(self, QuantizedDType::External(_))
            }


        }
        
        // ==================== External Type Support ====================
        
        /// External type operations
        /// 
        /// All operations available to built-in types:
        /// - CPU ops (required): quantize, dequantize, storage_size, matmul
        /// - CUDA ops (optional): dequantize, matmul
        /// - Metal ops (optional): dequantize, matmul
        pub struct ExternalQuantOps {
            // Static size per element (bytes)
            pub size_in_bytes: usize,
            
            // CPU operations (required)
            pub quantize_cpu: fn(&[f32]) -> crate::Result<Vec<u8>>,
            pub dequantize_cpu: fn(&[u8], &mut [f32]) -> crate::Result<()>,
            pub storage_size_in_bytes: fn(usize) -> usize,
            pub matmul_cpu: fn(&[f32], &[usize], &[u8], &[usize]) -> crate::Result<Vec<f32>>,
            
            // CUDA ops (optional)
            #[cfg(feature = "cuda")]
            pub dequantize_cuda: Option<fn(&cudarc::driver::CudaSlice<u8>, &mut cudarc::driver::CudaSlice<f32>) -> crate::Result<()>>,
            #[cfg(feature = "cuda")]
            pub matmul_cuda: Option<fn(&cudarc::driver::CudaSlice<u8>, &[usize], &cudarc::driver::CudaSlice<u8>, &[usize]) -> crate::Result<cudarc::driver::CudaSlice<u8>>>,
            
            // Metal ops (optional)
            #[cfg(feature = "metal")]
            pub dequantize_metal: Option<fn(&metal::Buffer, &mut metal::Buffer) -> crate::Result<()>>,
            #[cfg(feature = "metal")]
            pub matmul_metal: Option<fn(&metal::Buffer, &[usize], &metal::Buffer, &[usize]) -> crate::Result<metal::Buffer>>,
        }
        
        /// Global registry for external types (thread-safe)
        static EXTERNAL_TYPE_REGISTRY: std::sync::OnceLock<std::sync::RwLock<std::collections::HashMap<&'static str, ExternalQuantOps>>> 
            = std::sync::OnceLock::new();
        
        /// Register external quantized type by name
        /// 
        /// External types provide same operations as built-in types.
        /// 
        /// # Panics
        /// If type name already registered
        pub fn register_external_quant_type(
            name: &'static str,
            ops: ExternalQuantOps,
        ) -> QuantizedDType {
            let registry = EXTERNAL_TYPE_REGISTRY.get_or_init(|| {
                std::sync::RwLock::new(std::collections::HashMap::new())
            });
            
            let mut map = registry.write().unwrap();
            
            if map.contains_key(name) {
                panic!("External quantized type '{}' is already registered", name);
            }
            
            map.insert(name, ops);
            
            QuantizedDType::External(name)
        }
        
        /// Find external type by name
        pub fn find_external_type_by_name(name: &'static str) -> Option<QuantizedDType> {
            let registry = EXTERNAL_TYPE_REGISTRY.get()?;
            let map = registry.read().unwrap();
            
            if map.contains_key(name) {
                Some(QuantizedDType::External(name))
            } else {
                None
            }
        }
        
        
        /// Dispatch functions: compile-time for built-in, HashMap lookup for external
        pub mod quantized_dispatch {
            use super::*;
            
            // ==================== CPU Backend ====================
            
            #[inline]
            pub fn dequantize_cpu(
                id: QuantizedDType,
                data: &[u8],
                output: &mut [f32]
            ) -> crate::Result<()> {
                match id {
                    // Built-in: compile-time dispatch
                    #(#dequantize_arms,)*
                    
                    // External: HashMap lookup
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let map = registry.read().unwrap();
                        let ops = map.get(name)
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        
                        (ops.dequantize_cpu)(data, output)
                    }
                }
            }
            
            #[inline]
            pub fn quantize_cpu(
                id: QuantizedDType,
                input: &[f32]
            ) -> crate::Result<Vec<u8>> {
                match id {
                    #(#quantize_arms,)*
                    
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let map = registry.read().unwrap();
                        let ops = map.get(name)
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        
                        (ops.quantize_cpu)(input)
                    }
                }
            }
            
            #[inline]
            pub fn storage_size_in_bytes(
                id: QuantizedDType,
                num_elements: usize
            ) -> usize {
                match id {
                    #(#storage_size_arms,)*
                    
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get();
                        if let Some(registry) = registry {
                            let map = registry.read().unwrap();
                            if let Some(ops) = map.get(name) {
                                return (ops.storage_size_in_bytes)(num_elements);
                            }
                        }
                        0
                    }
                }
            }
            
            /// Matrix multiplication: f32 × quantized → f32 (mixed precision only)
            /// Pattern: f32_activations @ quantized_weights (common in inference)
            /// Other combinations (quantized × f32, quantized × quantized) auto-dequantize
            #[inline]
            pub fn matmul_cpu(
                lhs_f32: &[f32],
                lhs_shape: &[usize],
                rhs_id: QuantizedDType,
                rhs_data: &[u8],
                rhs_shape: &[usize],
            ) -> crate::Result<Vec<f32>> {
                match rhs_id {
                    #(#matmul_arms,)*
                    
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let map = registry.read().unwrap();
                        let ops = map.get(name)
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        
                        (ops.matmul_cpu)(lhs_f32, lhs_shape, rhs_data, rhs_shape)
                    }
                }
            }
            
            // ==================== CUDA Backend ====================
            
            #[cfg(feature = "cuda")]
            #[inline]
            pub fn dequantize_cuda(
                id: QuantizedDType,
                data: &cudarc::driver::CudaSlice<u8>,
                output: &mut cudarc::driver::CudaSlice<f32>
            ) -> crate::Result<()> {
                match id {
                    #(
                        QuantizedDType::#type_names => {
                            // Try CUDA implementation, fallback to CPU
                            #type_names::dequantize_cuda(data, output)
                                .or_else(|_| {
                                    let cpu_data = data.to_host()?;
                                    let mut cpu_output = vec![0.0f32; output.len()];
                                    #type_names::dequantize_cpu(&cpu_data, &mut cpu_output)?;
                                    output.copy_from_host(&cpu_output)?;
                                    Ok(())
                                })
                        }
                    )*
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let map = registry.read().unwrap();
                        let ops = map.get(name)
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        
                        // Try CUDA, fallback to CPU
                        if let Some(cuda_fn) = ops.dequantize_cuda {
                            cuda_fn(data, output)
                        } else {
                            let cpu_data = data.to_host()?;
                            let mut cpu_output = vec![0.0f32; output.len()];
                            (ops.dequantize_cpu)(&cpu_data, &mut cpu_output)?;
                            output.copy_from_host(&cpu_output)?;
                            Ok(())
                        }
                    }
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
            ) -> crate::Result<cudarc::driver::CudaSlice<u8>> {
                match id {
                    #(
                        QuantizedDType::#type_names => {
                            #type_names::matmul_cuda(lhs_data, lhs_shape, rhs_data, rhs_shape)
                        }
                    )*
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let map = registry.read().unwrap();
                        let ops = map.get(name)
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        
                        // CUDA implementation or error
                        if let Some(cuda_fn) = ops.matmul_cuda {
                            cuda_fn(lhs_data, lhs_shape, rhs_data, rhs_shape)
                        } else {
                            Err(crate::Error::Msg(format!("External type '{}' does not have CUDA matmul implementation", name)))
                        }
                    }
                }
            }
            
            // ==================== Metal Backend ====================
            
            #[cfg(feature = "metal")]
            #[inline]
            pub fn dequantize_metal(
                id: QuantizedDType,
                data: &metal::Buffer,
                output: &mut metal::Buffer
            ) -> crate::Result<()> {
                match id {
                    #(
                        QuantizedDType::#type_names => {
                            // Try Metal implementation directly
                            #type_names::dequantize_metal(data, output)
                        }
                    )*
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let map = registry.read().unwrap();
                        let ops = map.get(name)
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        
                        // Try Metal or error
                        if let Some(metal_fn) = ops.dequantize_metal {
                            metal_fn(data, output)
                        } else {
                            Err(crate::Error::Msg(format!("External type '{}' does not have Metal dequantize implementation", name)))
                        }
                    }
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
            ) -> crate::Result<metal::Buffer> {
                match id {
                    #(
                        QuantizedDType::#type_names => {
                            #type_names::matmul_metal(lhs_data, lhs_shape, rhs_data, rhs_shape)
                        }
                    )*
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let map = registry.read().unwrap();
                        let ops = map.get(name)
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        
                        // Metal implementation or error
                        if let Some(metal_fn) = ops.matmul_metal {
                            metal_fn(lhs_data, lhs_shape, rhs_data, rhs_shape)
                        } else {
                            Err(crate::Error::Msg(format!("External type '{}' does not have Metal matmul implementation", name)))
                        }
                    }
                }
            }
        }
        
        // Get type name helper
        #[inline]
        pub fn get_quantized_name(id: QuantizedDType) -> &'static str {
            id.name()
        }
    };
    
    TokenStream::from(expanded)
}

/// Register external quantized type with simplified API
/// 
/// Type must implement QuantizedType trait and provide all required methods:
/// - CPU ops (required): dequantize, quantize, storage_size_in_bytes, matmul
/// - CUDA ops (optional): dequantize_cuda, matmul_cuda
/// - Metal ops (optional): dequantize_metal, matmul_metal
/// 
/// Example:
/// ```ignore
/// #[derive(QuantizedType)]
/// #[quantized(name = "my_q4", size_in_bytes = 1)]
/// pub struct MyQ4;
/// 
/// impl MyQ4 {
///     pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<()> { /* ... */ }
///     pub fn quantize(input: &[f32]) -> Result<Vec<u8>> { /* ... */ }
///     pub fn storage_size_in_bytes(n: usize) -> usize { /* ... */ }
///     pub fn matmul(lhs: &[f32], lhs_shape: &[usize], rhs: &[u8], rhs_shape: &[usize]) 
///         -> Result<Vec<f32>> { /* ... */ }
/// }
/// 
/// register_external_quantized_type!(MyQ4);
/// ```
#[proc_macro]
pub fn register_external_quantized_type(input: TokenStream) -> TokenStream {
    let type_name = parse_macro_input!(input as syn::Ident);
    
    let expanded = quote! {
        /// Get QuantizedDType for this external type
        /// 
        /// Registers type on first call, safe to call multiple times.
        pub fn get_quantized_dtype() -> candle_core::quantized::QuantizedDType {
            use std::sync::OnceLock;
            use candle_core::quantized::*;
            
            static DTYPE: OnceLock<QuantizedDType> = OnceLock::new();
            
            *DTYPE.get_or_init(|| {
                let ops = ExternalQuantOps {
                    size_in_bytes: <#type_name as candle_core::dtype::QuantizedType>::SIZE_IN_BYTES,
                    quantize_cpu: #type_name::quantize,
                    dequantize_cpu: #type_name::dequantize,
                    storage_size_in_bytes: #type_name::storage_size_in_bytes,
                    matmul_cpu: #type_name::matmul,
                    
                    #[cfg(feature = "cuda")]
                    dequantize_cuda: {
                        #[cfg(feature = "cuda")]
                        { Some(#type_name::dequantize_cuda) }
                        #[cfg(not(feature = "cuda"))]
                        { None }
                    },
                    
                    #[cfg(feature = "cuda")]
                    matmul_cuda: {
                        #[cfg(feature = "cuda")]
                        { Some(#type_name::matmul_cuda) }
                        #[cfg(not(feature = "cuda"))]
                        { None }
                    },
                    
                    #[cfg(feature = "metal")]
                    dequantize_metal: {
                        #[cfg(feature = "metal")]
                        { Some(#type_name::dequantize_metal) }
                        #[cfg(not(feature = "metal"))]
                        { None }
                    },
                    
                    #[cfg(feature = "metal")]
                    matmul_metal: {
                        #[cfg(feature = "metal")]
                        { Some(#type_name::matmul_metal) }
                        #[cfg(not(feature = "metal"))]
                        { None }
                    },
                };
                
                register_external_quant_type(
                    <#type_name as candle_core::dtype::QuantizedType>::NAME,
                    ops
                )
            })
        }
    };
    
    TokenStream::from(expanded)
}
