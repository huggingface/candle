use proc_macro::TokenStream;
use quote::quote;
use syn::{parse::Parser, parse_macro_input};

/// Register multiple quantized types and generate dispatch code with automatic backend detection.
///
/// # Overview
///
/// This macro generates a complete quantized type system with:
/// - `QuantizedDType` enum with all registered types plus external type support
/// - Automatic compile-time backend detection (CPU/CUDA/Metal) per type
/// - Zero-cost dispatch functions that are feature-gated
/// - External type registration system via HashMap
///
/// ## Example
///
/// ```ignore
/// // Type with CPU support only
/// impl QuantizedType for GgmlQ4_0 { ... }
/// impl QuantizedCpuOps for GgmlQ4_0 { ... }
/// // No QuantizedCudaOps impl
///
/// // This will compile and work:
/// dequantize_cpu(QuantizedDType::GgmlQ4_0, data, output)?;
///
/// // This will return an error at runtime (not available):
/// dequantize_cuda(QuantizedDType::GgmlQ4_0, data, output)?;
///
/// // This is detected at compile time:
/// assert_eq!(QuantizedDType::GgmlQ4_0.has_cpu(), true);   // Known at compile time
/// assert_eq!(QuantizedDType::GgmlQ4_0.has_cuda(), false); // Known at compile time
/// ```
///
/// # Usage
///
/// ```ignore
/// register_quantized_types! {
///     GgmlQ2K, GgmlQ3K, GgmlQ4K, GgmlQ5K, GgmlQ6K, GgmlQ8K,
///     GgmlQ4_0, GgmlQ4_1, GgmlQ5_0, GgmlQ5_1, GgmlQ8_0, GgmlQ8_1
/// }
/// ```
///
/// Each type must implement `QuantizedType` and optionally `QuantizedCpuOps`,
/// `QuantizedCudaOps` (with `cuda` feature), and/or `QuantizedMetalOps` (with `metal` feature).
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

    // Generate match arms for get_name
    let name_arms = type_names.iter().map(|name| {
        quote! {
            QuantizedDType::#name => #name::NAME
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
                        EXTERNAL_TYPE_REGISTRY.get()
                            .and_then(|r| r.read().ok())
                            .and_then(|m| m.get(name).map(|impl_| impl_.size_in_bytes))
                            .unwrap_or(1)
                    }
                }
            }

            /// Calculate storage size in bytes for given number of elements
            #[inline]
            pub fn storage_size_in_bytes(self, num_elements: usize) -> usize {
                quantized_dispatch::storage_size_in_bytes(self, num_elements)
            }

            /// Infer number of f32 elements from quantized data size
            #[inline]
            pub fn infer_element_count(self, data_len: usize) -> usize {
                quantized_dispatch::infer_element_count(self, data_len)
            }

            /// Check if external type
            #[inline]
            pub const fn is_external(self) -> bool {
                matches!(self, QuantizedDType::External(_))
            }

            /// Check if CPU operations are supported
            ///
            /// Detects at compile-time whether a type implements `QuantizedCpuOps`.
            /// Uses autoderef-based specialization for stable Rust compatibility.
            /// For external types, always returns true (they must provide CPU ops).
            #[inline]
            pub fn has_cpu(self) -> bool {
                // Helper trait for compile-time detection using autoderef
                trait CpuCheck {
                    fn has_cpu(&self) -> bool;
                }

                // Level 1 (fallback): No CPU support - requires 2 auto-derefs
                struct Wrap<T>(std::marker::PhantomData<T>);
                impl<T: candle_macros_types::QuantizedType> CpuCheck for Wrap<T> {
                    #[inline(always)]
                    fn has_cpu(&self) -> bool {
                        false
                    }
                }

                // Level 0 (specialized): Has CPU support - requires 1 auto-deref
                // Blanket impl for types with QuantizedCpuOps
                impl<T: candle_macros_types::QuantizedCpuOps> CpuCheck for &Wrap<T> {
                    #[inline(always)]
                    fn has_cpu(&self) -> bool {
                        true
                    }
                }

                // Use autoderef directly on concrete types in match arms
                // If Type: QuantizedCpuOps, (&&Wrap<Type>).has_cpu() derefs once to &Wrap<Type> (returns true)
                // Otherwise, (&&Wrap<Type>).has_cpu() derefs twice to Wrap<Type> (returns false)
                match self {
                    #(QuantizedDType::#type_names => (&&Wrap::<#type_names>(std::marker::PhantomData)).has_cpu(),)*
                    QuantizedDType::External(name) => {
                        EXTERNAL_TYPE_REGISTRY.get()
                            .and_then(|r| r.read().ok())
                            .and_then(|m| m.get(name).map(|impl_|
                                impl_.cpu_quantize.is_some() || impl_.cpu_dequantize.is_some() || impl_.cpu_matmul.is_some()
                            ))
                            .unwrap_or(false)
                    }
                }
            }

            /// Check if CUDA operations are supported
            ///
            /// Detects at compile-time whether a type implements `QuantizedCudaOps`.
            /// Uses autoderef-based specialization for stable Rust compatibility.
            /// For external types, checks if CUDA ops were registered.
            #[inline]
            pub fn has_cuda(self) -> bool {
                #[cfg(feature = "cuda")]
                {
                    // Helper macro to check if a type implements QuantizedCudaOps for any device
                    macro_rules! check_cuda {
                        ($ty:ty) => {{
                            // Helper trait for compile-time detection using autoderef
                            trait CudaCheck {
                                fn check(&self) -> bool;
                            }

                            // Level 1 (fallback): No CUDA support - requires 2 auto-derefs
                            struct Wrap<T>(std::marker::PhantomData<T>);
                            impl<T> CudaCheck for Wrap<T>
                            where
                                T: candle_macros_types::QuantizedType,
                            {
                                #[inline(always)]
                                fn check(&self) -> bool {
                                    false
                                }
                            }

                            // Level 0 (specialized): Has CUDA support - requires 1 auto-deref
                            // This implementation is only valid if T: QuantizedCudaOps<D> for *some* D
                            // We use a concrete dummy device to test this
                            struct DummyDevice;
                            impl candle_macros_types::CudaStorageDevice for DummyDevice {
                                fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
                                    &self,
                                    _len: usize,
                                ) -> std::result::Result<cudarc::driver::CudaSlice<T>, Box<dyn std::error::Error + Send + Sync>> {
                                    unreachable!()
                                }
                                fn as_any(&self) -> &dyn std::any::Any {
                                    self
                                }
                            }

                            impl<T> CudaCheck for &Wrap<T>
                            where
                                T: candle_macros_types::QuantizedCudaOps<DummyDevice>,
                            {
                                #[inline(always)]
                                fn check(&self) -> bool {
                                    true
                                }
                            }

                            (&&Wrap::<$ty>(std::marker::PhantomData)).check()
                        }};
                    }

                    // Use the macro in match arms
                    match self {
                        #(QuantizedDType::#type_names => check_cuda!(#type_names),)*
                        QuantizedDType::External(_name) => {
                            false
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = self;
                    false
                }
            }

            /// Check if Metal operations are supported
            ///
            /// Detects at compile-time whether a type implements `QuantizedMetalOps`.
            /// Uses autoderef-based specialization for stable Rust compatibility.
            /// For external types, checks if Metal ops were registered.
            #[inline]
            pub fn has_metal(self) -> bool {
                #[cfg(feature = "metal")]
                {
                    // Helper trait for compile-time detection using autoderef
                    trait MetalCheck {
                        fn has_metal(&self) -> bool;
                    }

                    // Level 1 (fallback): No Metal support - requires 2 auto-derefs
                    struct Wrap<T>(std::marker::PhantomData<T>);
                    impl<T: candle_macros_types::QuantizedType> MetalCheck for Wrap<T> {
                        #[inline(always)]
                        fn has_metal(&self) -> bool {
                            false
                        }
                    }

                    // Level 0 (specialized): Has Metal support - requires 1 auto-deref
                    // Blanket impl for types with QuantizedMetalOps
                    impl<T: candle_macros_types::QuantizedMetalOps> MetalCheck for &Wrap<T> {
                        #[inline(always)]
                        fn has_metal(&self) -> bool {
                            true
                        }
                    }

                    // Helper function that uses autoderef to pick the right impl
                    // Use autoderef directly on concrete types in match arms
                    match self {
                        #(QuantizedDType::#type_names => (&&Wrap::<#type_names>(std::marker::PhantomData)).has_metal(),)*
                        QuantizedDType::External(name) => {
                            false
                        }
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    let _ = self;
                    false
                }
            }

            /// Convenience dequantize function returning a newly allocated vector
            /// If num_elements is None, infers number of elements from data size
            #[inline]
            pub fn dequantize(self, data: &[u8], num_elements: Option<usize>) -> crate::Result<Vec<f32>> {
                let mut output = if let Some(n) = num_elements {
                    vec![0.0f32; n]
                }else{
                    // Calculate number of elements from data size
                    // For quantized types, we need to know the block structure
                    let num_elements = quantized_dispatch::infer_element_count(self, data.len());
                    vec![0.0f32; num_elements]
                };
                quantized_dispatch::dequantize_cpu(self, data, &mut output)?;
                Ok(output)
            }

            /// Convenience quantize function
            #[inline]
            pub fn quantize(self, input: &[f32]) -> crate::Result<Vec<u8>> {
                quantized_dispatch::quantize_cpu(self, input)
            }

            /// CUDA dequantize function returning a newly allocated CudaSlice
            /// If num_elements is None, infers number of elements from data size
            #[cfg(feature = "cuda")]
            #[inline]
            pub fn dequantize_cuda<D>(
                self,
                data: &cudarc::driver::CudaSlice<u8>,
                device: &D,
                num_elements: Option<usize>
            ) -> crate::Result<cudarc::driver::CudaSlice<f32>>
            where
                D: candle_macros_types::CudaStorageDevice,
            {
                // Determine number of elements
                let num_elements = if let Some(n) = num_elements {
                    n
                } else {
                    // Infer from data size
                    quantized_dispatch::infer_element_count(self, data.len())
                };

                // Allocate output buffer on GPU
                let mut output = device.alloc_zeros::<f32>(num_elements)
                    .map_err(|e| crate::Error::Msg(e.to_string()))?;

                // Perform dequantization
                quantized_dispatch::dequantize_cuda(self, data, &mut output, device)?;

                Ok(output)
            }

            /// CUDA quantize function
            #[cfg(feature = "cuda")]
            #[inline]
            pub fn quantize_cuda<D>(
                self,
                input: &cudarc::driver::CudaSlice<f32>,
                device: &D
            ) -> crate::Result<cudarc::driver::CudaSlice<u8>>
            where
                D: candle_macros_types::CudaStorageDevice,
            {
                quantized_dispatch::quantize_cuda(self, input, device)
            }
        }

        // ==================== External Type Support ====================

        /// External type operations holder
        ///
        /// Stores function pointers for external types, allowing them to work exactly
        /// like built-in types through dynamic dispatch.
        struct ExternalTypeImpl {
            // Metadata (from QuantizedType associated consts)
            size_in_bytes: usize,

            // Base operations (from QuantizedType trait methods)
            storage_size_in_bytes: Box<dyn Fn(usize) -> usize + Send + Sync>,
            infer_element_count: Box<dyn Fn(usize) -> usize + Send + Sync>,

            // Optional CPU operations (function pointers extracted from trait methods)
            cpu_quantize: Option<Box<dyn Fn(&[f32]) -> std::result::Result<Vec<u8>, String> + Send + Sync>>,
            cpu_dequantize: Option<Box<dyn Fn(&[u8], &mut [f32]) -> std::result::Result<(), String> + Send + Sync>>,
            cpu_matmul: Option<Box<dyn Fn(&[f32], &[usize], &[u8], &[usize]) -> std::result::Result<Vec<f32>, String> + Send + Sync>>,
        }

        /// Global registry for external types (thread-safe)
        static EXTERNAL_TYPE_REGISTRY: std::sync::OnceLock<std::sync::RwLock<std::collections::HashMap<&'static str, ExternalTypeImpl>>>
            = std::sync::OnceLock::new();

        /// Register external quantized type by name
        ///
        /// External types work exactly like built-in types but use dynamic dispatch via closures.
        /// Must provide metadata and base operations, CPU/CUDA/Metal ops are optional.
        ///
        /// # Panics
        /// If type name already registered
        #[allow(clippy::too_many_arguments)]
        pub fn register_external_quant_type(
            name: &'static str,
            size_in_bytes: usize,
            storage_size_in_bytes: Box<dyn Fn(usize) -> usize + Send + Sync>,
            infer_element_count: Box<dyn Fn(usize) -> usize + Send + Sync>,
            cpu_quantize: Option<Box<dyn Fn(&[f32]) -> std::result::Result<Vec<u8>, String> + Send + Sync>>,
            cpu_dequantize: Option<Box<dyn Fn(&[u8], &mut [f32]) -> std::result::Result<(), String> + Send + Sync>>,
            cpu_matmul: Option<Box<dyn Fn(&[f32], &[usize], &[u8], &[usize]) -> std::result::Result<Vec<f32>, String> + Send + Sync>>,
        ) -> QuantizedDType {
            let registry = EXTERNAL_TYPE_REGISTRY.get_or_init(|| {
                std::sync::RwLock::new(std::collections::HashMap::new())
            });

            let mut map = registry.write().unwrap();

            if map.contains_key(name) {
                panic!("External quantized type '{}' is already registered", name);
            }

            map.insert(name, ExternalTypeImpl {
                size_in_bytes,
                storage_size_in_bytes,
                infer_element_count,
                cpu_quantize,
                cpu_dequantize,
                cpu_matmul,
            });

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

            #[inline]
            pub fn infer_element_count(id: QuantizedDType, data_len: usize) -> usize {
                match id {
                    #(QuantizedDType::#type_names => #type_names::default().infer_element_count(data_len),)*
                    QuantizedDType::External(name) => {
                        EXTERNAL_TYPE_REGISTRY.get()
                            .and_then(|r| r.read().ok())
                            .and_then(|m| m.get(name).map(|impl_| (impl_.infer_element_count)(data_len)))
                            .unwrap_or(data_len)
                    }
                }
            }

            #[inline]
            pub fn storage_size_in_bytes(id: QuantizedDType, num_elements: usize) -> usize {
                match id {
                    #(QuantizedDType::#type_names => #type_names::default().storage_size_in_bytes(num_elements),)*
                    QuantizedDType::External(name) => {
                        EXTERNAL_TYPE_REGISTRY.get()
                            .and_then(|r| r.read().ok())
                            .and_then(|m| m.get(name).map(|impl_| (impl_.storage_size_in_bytes)(num_elements)))
                            .unwrap_or(0)
                    }
                }
            }

            // Helper trait to provide compile-time dispatch for optional backends
            // Uses autoderef-based specialization for stable Rust
            #[inline]
            pub fn dequantize_cpu(id: QuantizedDType, data: &[u8], output: &mut [f32]) -> crate::Result<()> {
                match id {
                    #(QuantizedDType::#type_names => (&&CpuWrap::<#type_names>(std::marker::PhantomData)).try_dequantize_cpu(data, output),)*
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let map = registry.read().unwrap();
                        let impl_ = map.get(name)
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;

                        if let Some(ref func) = impl_.cpu_dequantize {
                            func(data, output).map_err(|e| crate::Error::Msg(e))
                        } else {
                            Err(crate::Error::Msg(format!("External type '{}' does not implement CPU dequantize", name)))
                        }
                    }
                }
            }

            #[inline]
            pub fn quantize_cpu(id: QuantizedDType, input: &[f32]) -> crate::Result<Vec<u8>> {
                match id {
                    #(QuantizedDType::#type_names => (&&CpuWrap::<#type_names>(std::marker::PhantomData)).try_quantize_cpu(input),)*
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let map = registry.read().unwrap();
                        let impl_ = map.get(name)
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;

                        if let Some(ref func) = impl_.cpu_quantize {
                            func(input).map_err(|e| crate::Error::Msg(e))
                        } else {
                            Err(crate::Error::Msg(format!("External type '{}' does not implement CPU quantize", name)))
                        }
                    }
                }
            }

            #[inline]
            pub fn matmul_cpu(
                lhs_f32: &[f32],
                lhs_shape: &[usize],
                rhs_id: QuantizedDType,
                rhs_data: &[u8],
                rhs_shape: &[usize],
            ) -> crate::Result<Vec<f32>> {
                match rhs_id {
                    #(QuantizedDType::#type_names => (&&CpuWrap::<#type_names>(std::marker::PhantomData)).try_matmul_cpu(lhs_f32, lhs_shape, rhs_data, rhs_shape),)*
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let map = registry.read().unwrap();
                        let impl_ = map.get(name)
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;

                        if let Some(ref func) = impl_.cpu_matmul {
                            func(lhs_f32, lhs_shape, rhs_data, rhs_shape).map_err(|e| crate::Error::Msg(e))
                        } else {
                            Err(crate::Error::Msg(format!("External type '{}' does not implement CPU matmul", name)))
                        }
                    }
                }
            }

            // ============================================================================
            // CPU Dispatch - Autoderef-Based Specialization
            // ============================================================================
            //
            // This uses stable Rust to achieve compile-time backend detection without
            // specialization features. Based on:
            // https://lukaskalbertodt.github.io/2019/12/05/generalized-autoref-based-specialization.html
            //
            // How it works:
            // 1. We create a wrapper type CpuWrap<T> to avoid blanket impl interference
            // 2. We implement MaybeCpuOps at two levels:
            //    - Level 1 (this impl): For CpuWrap<T> - requires 2 auto-derefs
            //    - Level 0 (below): For &CpuWrap<T> where T: QuantizedCpuOps - requires 1 auto-deref
            // 3. Method calls use (&&CpuWrap::<Type>(PhantomData)).method()
            // 4. Rust's method resolution prefers fewer auto-derefs, so:
            //    - If Type implements QuantizedCpuOps: Level 0 is chosen (1 deref)
            //    - Otherwise: Level 1 is chosen (2 derefs) and returns an error
            //
            // This decision happens at compile time with zero runtime cost!
            //
            struct CpuWrap<T>(std::marker::PhantomData<T>);

            pub trait MaybeCpuOps {
                fn try_dequantize_cpu(&self, data: &[u8], output: &mut [f32]) -> crate::Result<()>;
                fn try_quantize_cpu(&self, input: &[f32]) -> crate::Result<Vec<u8>>;
                fn try_matmul_cpu(&self, lhs: &[f32], lhs_shape: &[usize], rhs: &[u8], rhs_shape: &[usize]) -> crate::Result<Vec<f32>>;
            }

            // Level 1 (fallback): For types WITHOUT CPU support - requires 2 auto-derefs
            // This impl is chosen when no more specific impl is available
            impl<T> MaybeCpuOps for CpuWrap<T>
            where
                T: candle_macros_types::QuantizedType,
            {
                #[inline(always)]
                fn try_dequantize_cpu(&self, _data: &[u8], _output: &mut [f32]) -> crate::Result<()> {
                    Err(crate::Error::Msg(format!("Type '{}' does not implement CPU dequantize", T::NAME)))
                }

                #[inline(always)]
                fn try_quantize_cpu(&self, _input: &[f32]) -> crate::Result<Vec<u8>> {
                    Err(crate::Error::Msg(format!("Type '{}' does not implement CPU quantize", T::NAME)))
                }

                #[inline(always)]
                fn try_matmul_cpu(&self, _lhs: &[f32], _lhs_shape: &[usize], _rhs: &[u8], _rhs_shape: &[usize]) -> crate::Result<Vec<f32>> {
                    Err(crate::Error::Msg(format!("Type '{}' does not implement CPU matmul", T::NAME)))
                }
            }

            // Level 0 (specialized): Blanket impl for types WITH CPU support - requires only 1 auto-deref
            // This impl is preferred over Level 1 because method resolution favors impls requiring fewer auto-derefs
            //
            // On stable Rust, we use a blanket impl with QuantizedCpuOps constraint.
            // This works because:
            // 1. The blanket impl exists for ALL types that implement QuantizedCpuOps + QuantizedType
            // 2. For types that don't implement QuantizedCpuOps, this impl doesn't exist, so Level 1 is used
            // 3. No explicit per-type impls needed - it's automatic!
            impl<T> MaybeCpuOps for &CpuWrap<T>
            where
                T: candle_macros_types::QuantizedCpuOps + Default,
            {
                #[inline(always)]
                fn try_dequantize_cpu(&self, data: &[u8], output: &mut [f32]) -> crate::Result<()> {
                    T::default().dequantize(data, output).map_err(|e| crate::Error::Msg(e))
                }

                #[inline(always)]
                fn try_quantize_cpu(&self, input: &[f32]) -> crate::Result<Vec<u8>> {
                    T::default().quantize(input).map_err(|e| crate::Error::Msg(e))
                }

                #[inline(always)]
                fn try_matmul_cpu(&self, lhs: &[f32], lhs_shape: &[usize], rhs: &[u8], rhs_shape: &[usize]) -> crate::Result<Vec<f32>> {
                    T::default().matmul(lhs, lhs_shape, rhs, rhs_shape).map_err(|e| crate::Error::Msg(e))
                }
            }

            // ============================================================================
            // CUDA Dispatch - Autoderef-Based Specialization (same pattern as CPU)
            // ============================================================================
            #[cfg(feature = "cuda")]
            struct CudaWrap<T>(std::marker::PhantomData<T>);

            #[cfg(feature = "cuda")]
            pub trait MaybeCudaOps<D>
            where
                D: candle_macros_types::CudaStorageDevice
            {
                fn try_dequantize_cuda(&self, data: &cudarc::driver::CudaSlice<u8>, output: &mut cudarc::driver::CudaSlice<f32>, device: &D) -> crate::Result<()>;
                fn try_quantize_cuda(&self, input: &cudarc::driver::CudaSlice<f32>, device: &D) -> crate::Result<cudarc::driver::CudaSlice<u8>>;
                fn try_matmul_cuda(&self, lhs: &cudarc::driver::CudaSlice<f32>, lhs_shape: &[usize], rhs: &cudarc::driver::CudaSlice<u8>, rhs_shape: &[usize], device: &D) -> crate::Result<cudarc::driver::CudaSlice<f32>>;
            }

            // Level 1 (fallback): For types WITHOUT CUDA support - requires 2 auto-derefs
            #[cfg(feature = "cuda")]
            impl<T, D> MaybeCudaOps<D> for CudaWrap<T>
            where
                T: candle_macros_types::QuantizedType,
                D: candle_macros_types::CudaStorageDevice,
            {
                #[inline(always)]
                fn try_dequantize_cuda(&self, _data: &cudarc::driver::CudaSlice<u8>, _output: &mut cudarc::driver::CudaSlice<f32>, _device: &D) -> crate::Result<()> {
                    Err(crate::Error::Msg(format!("Type '{}' does not implement CUDA dequantize", T::NAME)))
                }

                #[inline(always)]
                fn try_quantize_cuda(&self, _input: &cudarc::driver::CudaSlice<f32>, _device: &D) -> crate::Result<cudarc::driver::CudaSlice<u8>> {
                    Err(crate::Error::Msg(format!("Type '{}' does not implement CUDA quantize", T::NAME)))
                }

                #[inline(always)]
                fn try_matmul_cuda(&self, _lhs: &cudarc::driver::CudaSlice<f32>, _lhs_shape: &[usize], _rhs: &cudarc::driver::CudaSlice<u8>, _rhs_shape: &[usize], _device: &D) -> crate::Result<cudarc::driver::CudaSlice<f32>> {
                    Err(crate::Error::Msg(format!("Type '{}' does not implement CUDA matmul", T::NAME)))
                }
            }

            // Level 0 (specialized): Blanket impl for types WITH CUDA support - requires only 1 auto-deref
            // This impl is preferred over Level 1 because method resolution favors impls requiring fewer auto-derefs
            // Blanket impl automatically applies to any type implementing QuantizedCudaOps
            #[cfg(feature = "cuda")]
            impl<T, D> MaybeCudaOps<D> for &CudaWrap<T>
            where
                T: candle_macros_types::QuantizedCudaOps<D> + Default,
                D: candle_macros_types::CudaStorageDevice,
            {
                #[inline(always)]
                fn try_dequantize_cuda(&self, data: &cudarc::driver::CudaSlice<u8>, output: &mut cudarc::driver::CudaSlice<f32>, device: &D) -> crate::Result<()> {
                    T::default().dequantize_cuda(data, output, device).map_err(|e| crate::Error::Msg(e))
                }

                #[inline(always)]
                fn try_quantize_cuda(&self, input: &cudarc::driver::CudaSlice<f32>, device: &D) -> crate::Result<cudarc::driver::CudaSlice<u8>> {
                    T::default().quantize_cuda(input, device).map_err(|e| crate::Error::Msg(e))
                }

                #[inline(always)]
                fn try_matmul_cuda(&self, lhs: &cudarc::driver::CudaSlice<f32>, lhs_shape: &[usize], rhs: &cudarc::driver::CudaSlice<u8>, rhs_shape: &[usize], device: &D) -> crate::Result<cudarc::driver::CudaSlice<f32>> {
                    T::default().matmul_cuda(lhs, lhs_shape, rhs, rhs_shape, device).map_err(|e| crate::Error::Msg(e))
                }
            }

            #[cfg(feature = "cuda")]
            #[inline]
            pub fn dequantize_cuda<D: candle_macros_types::CudaStorageDevice>(
                id: QuantizedDType,
                data: &cudarc::driver::CudaSlice<u8>,
                output: &mut cudarc::driver::CudaSlice<f32>,
                device: &D,
            ) -> crate::Result<()> {
                match id {
                    #(QuantizedDType::#type_names => (&&CudaWrap::<#type_names>(std::marker::PhantomData)).try_dequantize_cuda(data, output, device),)*
                    QuantizedDType::External(name) => {
                         Err(crate::Error::Msg("Cuda currently not supported for external types".to_string()))
                    }
                }
            }

            #[cfg(feature = "cuda")]
            #[inline]
            pub fn quantize_cuda<D: candle_macros_types::CudaStorageDevice>(
                id: QuantizedDType,
                input: &cudarc::driver::CudaSlice<f32>,
                device: &D,
            ) -> crate::Result<cudarc::driver::CudaSlice<u8>> {
                match id {
                    #(QuantizedDType::#type_names => (&&CudaWrap::<#type_names>(std::marker::PhantomData)).try_quantize_cuda(input, device),)*
                    QuantizedDType::External(name) => {
                         Err(crate::Error::Msg("Cuda currently not supported for external types".to_string()))
                    }
                }
            }

            #[cfg(feature = "cuda")]
            #[inline]
            pub fn matmul_cuda<D: candle_macros_types::CudaStorageDevice>(
                id: QuantizedDType,
                lhs_data: &cudarc::driver::CudaSlice<f32>,
                lhs_shape: &[usize],
                rhs_data: &cudarc::driver::CudaSlice<u8>,
                rhs_shape: &[usize],
                device: &D,
            ) -> crate::Result<cudarc::driver::CudaSlice<f32>> {
                match id {
                    #(QuantizedDType::#type_names => (&&CudaWrap::<#type_names>(std::marker::PhantomData)).try_matmul_cuda(lhs_data, lhs_shape, rhs_data, rhs_shape, device),)*
                    QuantizedDType::External(name) => {
                        Err(crate::Error::Msg("Cuda currently not supported for external types".to_string()))
                    }
                }
            }

            // ============================================================================
            // Metal Dispatch - Autoderef-Based Specialization (same pattern as CPU/CUDA)
            // ============================================================================
            #[cfg(feature = "metal")]
            struct MetalWrap<T>(std::marker::PhantomData<T>);

            #[cfg(feature = "metal")]
            pub trait MaybeMetalOps {
                fn try_dequantize_metal(&self, data: &metal::Buffer, output: &mut metal::Buffer) -> crate::Result<()>;
                fn try_matmul_metal(&self, lhs: &metal::Buffer, lhs_shape: &[usize], rhs: &metal::Buffer, rhs_shape: &[usize]) -> crate::Result<metal::Buffer>;
            }

            // Level 1 (fallback): For types WITHOUT Metal support - requires 2 auto-derefs
            #[cfg(feature = "metal")]
            impl<T> MaybeMetalOps for MetalWrap<T>
            where
                T: candle_macros_types::QuantizedType,
            {
                #[inline(always)]
                fn try_dequantize_metal(&self, _data: &metal::Buffer, _output: &mut metal::Buffer) -> crate::Result<()> {
                    Err(crate::Error::Msg(format!("Type '{}' does not implement Metal dequantize", T::NAME)))
                }

                #[inline(always)]
                fn try_matmul_metal(&self, _lhs: &metal::Buffer, _lhs_shape: &[usize], _rhs: &metal::Buffer, _rhs_shape: &[usize]) -> crate::Result<metal::Buffer> {
                    Err(crate::Error::Msg(format!("Type '{}' does not implement Metal matmul", T::NAME)))
                }
            }

            // Level 0 (specialized): Blanket impl for types WITH Metal support - requires only 1 auto-deref
            // This impl is preferred over Level 1 because method resolution favors impls requiring fewer auto-derefs
            // Blanket impl automatically applies to any type implementing QuantizedMetalOps
            #[cfg(feature = "metal")]
            impl<T> MaybeMetalOps for &MetalWrap<T>
            where
                T: candle_macros_types::QuantizedMetalOps + Default,
            {
                #[inline(always)]
                fn try_dequantize_metal(&self, data: &metal::Buffer, output: &mut metal::Buffer) -> crate::Result<()> {
                    T::default().dequantize_metal(data, output).map_err(|e| crate::Error::Msg(e))
                }

                #[inline(always)]
                fn try_matmul_metal(&self, lhs: &metal::Buffer, lhs_shape: &[usize], rhs: &metal::Buffer, rhs_shape: &[usize]) -> crate::Result<metal::Buffer> {
                    T::default().matmul_metal(lhs, lhs_shape, rhs, rhs_shape).map_err(|e| crate::Error::Msg(e))
                }
            }

            #[cfg(feature = "metal")]
            #[inline]
            pub fn dequantize_metal(
                id: QuantizedDType,
                data: &metal::Buffer,
                output: &mut metal::Buffer
            ) -> crate::Result<()> {
                match id {
                    #(QuantizedDType::#type_names => (&&MetalWrap::<#type_names>(std::marker::PhantomData)).try_dequantize_metal(data, output),)*
                    QuantizedDType::External(name) => {
                        Err(crate::Error::Msg("Metal currently not supported for external types".to_string()))
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
                    #(QuantizedDType::#type_names => (&&MetalWrap::<#type_names>(std::marker::PhantomData)).try_matmul_metal(lhs_data, lhs_shape, rhs_data, rhs_shape),)*
                    QuantizedDType::External(name) => {
                        Err(crate::Error::Msg("Metal currently not supported for external types".to_string()))
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
/// Type must implement QuantizedType. CPU, CUDA, and Metal ops are all optional.
/// Implement QuantizedCpuOps, QuantizedCudaOps, and/or QuantizedMetalOps as needed.
/// Unimplemented operations will return errors when called.
///
/// Example:
/// ```ignore
/// use candle_macros_types::{QuantizedType, QuantizedCpuOps};
///
/// #[derive(Default)]
/// pub struct MyQ4;
///
/// impl QuantizedType for MyQ4 {
///     const NAME: &'static str = "my_q4";
///     const SIZE_IN_BYTES: usize = 1;
///     
///     fn storage_size_in_bytes(&self, n: usize) -> usize { /* ... */ }
///     fn infer_element_count(&self, data_len: usize) -> usize { /* ... */ }
/// }
///
/// // CPU ops are optional
/// impl QuantizedCpuOps for MyQ4 {
///     fn dequantize(&self, data: &[u8], output: &mut [f32]) -> Result<(), String> { /* ... */ }
///     fn quantize(&self, input: &[f32]) -> Result<Vec<u8>, String> { /* ... */ }
///     fn matmul(&self, lhs: &[f32], lhs_shape: &[usize], rhs: &[u8], rhs_shape: &[usize])
///         -> Result<Vec<f32>, String> { /* ... */ }
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
            use candle_macros_types::QuantizedType;

            static DTYPE: OnceLock<QuantizedDType> = OnceLock::new();

            *DTYPE.get_or_init(|| {
                // Helper trait for compile-time detection of CPU ops using autoderef
                trait CheckCpuOps<T> {
                    fn check_cpu_quantize(&self) -> Option<Box<dyn Fn(&[f32]) -> std::result::Result<Vec<u8>, String> + Send + Sync>>;
                    fn check_cpu_dequantize(&self) -> Option<Box<dyn Fn(&[u8], &mut [f32]) -> std::result::Result<(), String> + Send + Sync>>;
                    fn check_cpu_matmul(&self) -> Option<Box<dyn Fn(&[f32], &[usize], &[u8], &[usize]) -> std::result::Result<Vec<f32>, String> + Send + Sync>>;
                }

                // Level 1 (fallback): No CPU ops - requires 2 auto-derefs
                struct Wrap<T>(std::marker::PhantomData<T>);
                impl<T: QuantizedType + Default> CheckCpuOps<T> for Wrap<T> {
                    #[inline(always)]
                    fn check_cpu_quantize(&self) -> Option<Box<dyn Fn(&[f32]) -> Result<Vec<u8>, String> + Send + Sync>> {
                        None
                    }
                    #[inline(always)]
                    fn check_cpu_dequantize(&self) -> Option<Box<dyn Fn(&[u8], &mut [f32]) -> Result<(), String> + Send + Sync>> {
                        None
                    }
                    #[inline(always)]
                    fn check_cpu_matmul(&self) -> Option<Box<dyn Fn(&[f32], &[usize], &[u8], &[usize]) -> Result<Vec<f32>, String> + Send + Sync>> {
                        None
                    }
                }

                // Level 0 (specialized): Has CPU ops - requires 1 auto-deref
                impl<T: candle_macros_types::QuantizedCpuOps + Default + 'static> CheckCpuOps<T> for &Wrap<T> {
                    #[inline(always)]
                    fn check_cpu_quantize(&self) -> Option<Box<dyn Fn(&[f32]) -> std::result::Result<Vec<u8>, String> + Send + Sync>> {
                        Some(Box::new(|input: &[f32]| -> std::result::Result<Vec<u8>, String> {
                            T::default().quantize(input)
                        }))
                    }
                    #[inline(always)]
                    fn check_cpu_dequantize(&self) -> Option<Box<dyn Fn(&[u8], &mut [f32]) -> std::result::Result<(), String> + Send + Sync>> {
                        Some(Box::new(|data: &[u8], output: &mut [f32]| -> std::result::Result<(), String> {
                            T::default().dequantize(data, output)
                        }))
                    }
                    #[inline(always)]
                    fn check_cpu_matmul(&self) -> Option<Box<dyn Fn(&[f32], &[usize], &[u8], &[usize]) -> std::result::Result<Vec<f32>, String> + Send + Sync>> {
                        Some(Box::new(|lhs: &[f32], lhs_shape: &[usize], rhs: &[u8], rhs_shape: &[usize]| -> std::result::Result<Vec<f32>, String> {
                            T::default().matmul(lhs, lhs_shape, rhs, rhs_shape)
                        }))
                    }
                }

                let cpu_quantize = (&&Wrap::<#type_name>(std::marker::PhantomData)).check_cpu_quantize();
                let cpu_dequantize = (&&Wrap::<#type_name>(std::marker::PhantomData)).check_cpu_dequantize();
                let cpu_matmul = (&&Wrap::<#type_name>(std::marker::PhantomData)).check_cpu_matmul();



                // Create closures for base operations (QuantizedType trait methods)
                // These capture the type via its Default implementation
                let storage_size_fn: Box<dyn Fn(usize) -> usize + Send + Sync> = Box::new(|num_elements| {
                    #type_name::default().storage_size_in_bytes(num_elements)
                });

                let infer_count_fn: Box<dyn Fn(usize) -> usize + Send + Sync> = Box::new(|data_len| {
                    #type_name::default().infer_element_count(data_len)
                });

                register_external_quant_type(
                    <#type_name as candle_macros_types::QuantizedType>::NAME,
                    <#type_name as candle_macros_types::QuantizedType>::SIZE_IN_BYTES,
                    storage_size_fn,
                    infer_count_fn,
                    cpu_quantize,
                    cpu_dequantize,
                    cpu_matmul,
                )
            })
        }
    };

    TokenStream::from(expanded)
}
