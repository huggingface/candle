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
/// # Backend Detection via Autoderef-Based Specialization
///
/// **Important**: As of November 2025, Rust's `specialization` feature is still unstable.
/// This macro uses **autoderef-based specialization** as a stable alternative to achieve
/// compile-time backend detection without manual configuration.
///
/// The technique is based on:
/// ["Generalized Autoref-Based Specialization" by Lukas Kalbertodt](https://lukaskalbertodt.github.io/2019/12/05/generalized-autoref-based-specialization.html)
///
/// ## How It Works
///
/// For each backend (CPU, CUDA, Metal), we generate:
///
/// 1. **Wrapper Type**: `struct CpuWrap<T>(PhantomData<T>)`
///    - Avoids interference with other blanket implementations
///
/// 2. **Two-Level Trait Implementation**:
///    - **Level 1 (Fallback)**: `impl<T: QuantizedType> MaybeCpuOps for CpuWrap<T>`
///      - Returns errors for unsupported operations
///      - Requires 2 auto-derefs to call methods
///    - **Level 0 (Specialized)**: `impl MaybeCpuOps for &CpuWrap<Type> where Type: QuantizedCpuOps`
///      - Calls actual trait methods
///      - Requires only 1 auto-deref to call methods
///
/// 3. **Method Resolution**:
///    ```rust
///    (&&CpuWrap::<GgmlQ4_0>(PhantomData)).try_dequantize_cpu(data, output)
///    ```
///    - Rust's method resolution prefers methods requiring fewer auto-derefs
///    - If `GgmlQ4_0: QuantizedCpuOps`, the specialized impl (1 deref) is chosen
///    - Otherwise, the fallback impl (2 derefs) is chosen
///    - Decision happens at **compile time** with **zero runtime cost**
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
            pub const fn has_cpu(self) -> bool {
                // Autoderef-based specialization: wrapper type to avoid blanket impl interference
                struct Wrap<T>(std::marker::PhantomData<T>);

                // Helper trait for detection
                trait CpuCheck { const HAS: bool; }

                // Level 1 (fallback): No CPU support
                impl<T: candle_macros_types::QuantizedType> CpuCheck for Wrap<T> {
                    const HAS: bool = false;
                }

                // Level 0 (highest priority): Has CPU support if QuantizedCpuOps is implemented
                #(
                    impl CpuCheck for &Wrap<#type_names>
                    where #type_names: candle_macros_types::QuantizedCpuOps
                    {
                        const HAS: bool = true;
                    }
                )*

                match self {
                    #(QuantizedDType::#type_names => <&Wrap<#type_names> as CpuCheck>::HAS,)*
                    QuantizedDType::External(_) => true,
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
                    // Autoderef-based specialization: wrapper type to avoid blanket impl interference
                    struct Wrap<T>(std::marker::PhantomData<T>);

                    // Helper trait for detection
                    trait CudaCheck { const HAS: bool; }

                    // Level 1 (fallback): No CUDA support
                    impl<T: candle_macros_types::QuantizedType> CudaCheck for Wrap<T> {
                        const HAS: bool = false;
                    }

                    // Level 0 (highest priority): Has CUDA support if QuantizedCudaOps is implemented
                    #(
                        impl CudaCheck for &Wrap<#type_names>
                        where #type_names: candle_macros_types::QuantizedCudaOps
                        {
                            const HAS: bool = true;
                        }
                    )*

                    match self {
                        #(QuantizedDType::#type_names => <&Wrap<#type_names> as CudaCheck>::HAS,)*
                        QuantizedDType::External(name) => {
                            EXTERNAL_TYPE_REGISTRY.get()
                                .and_then(|r| r.read().ok())
                                .and_then(|m| m.get(name))
                                .map(|ops| ops.quantize_cuda.is_some())
                                .unwrap_or(false)
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
                    // Autoderef-based specialization: wrapper type to avoid blanket impl interference
                    struct Wrap<T>(std::marker::PhantomData<T>);

                    // Helper trait for detection
                    trait MetalCheck { const HAS: bool; }

                    // Level 1 (fallback): No Metal support
                    impl<T: candle_macros_types::QuantizedType> MetalCheck for Wrap<T> {
                        const HAS: bool = false;
                    }

                    // Level 0 (highest priority): Has Metal support if QuantizedMetalOps is implemented
                    #(
                        impl MetalCheck for &Wrap<#type_names>
                        where #type_names: candle_macros_types::QuantizedMetalOps
                        {
                            const HAS: bool = true;
                        }
                    )*

                    match self {
                        #(QuantizedDType::#type_names => <&&Wrap<#type_names> as MetalCheck>::HAS,)*
                        QuantizedDType::External(name) => {
                            EXTERNAL_TYPE_REGISTRY.get()
                                .and_then(|r| r.read().ok())
                                .and_then(|m| m.get(name))
                                .map(|ops| ops.dequantize_metal.is_some())
                                .unwrap_or(false)
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
            pub fn dequantize_cuda(
                self,
                data: &cudarc::driver::CudaSlice<u8>,
                device: &crate::CudaDevice,
                num_elements: Option<usize>
            ) -> crate::Result<cudarc::driver::CudaSlice<f32>> {
                // Determine number of elements
                let num_elements = if let Some(n) = num_elements {
                    n
                } else {
                    // Infer from data size
                    quantized_dispatch::infer_element_count(self, data.len())
                };

                // Allocate output buffer on GPU
                let mut output = device.alloc_zeros::<f32>(num_elements)?;

                // Perform dequantization
                quantized_dispatch::dequantize_cuda(self, data, &mut output)?;

                Ok(output)
            }

            /// CUDA quantize function
            #[cfg(feature = "cuda")]
            #[inline]
            pub fn quantize_cuda(
                self,
                input: &cudarc::driver::CudaSlice<f32>,
                device: &crate::CudaDevice
            ) -> crate::Result<cudarc::driver::CudaSlice<u8>> {
                let _ = device; // Keep for API compatibility but not needed for dispatch
                quantized_dispatch::quantize_cuda(self, input)
            }
        }

        // ==================== External Type Support ====================

        /// External type operations extracted from trait implementations
        ///
        /// All operations available to built-in types are function pointers
        /// extracted from types implementing the quantized traits.
        ///
        /// Note: GPU operations don't require device parameters as CudaSlice/Buffer
        /// already contain the device context.
        #[derive(Copy, Clone)]
        pub struct ExternalQuantOps {
            // Metadata
            pub size_in_bytes: usize,

            // CPU operations (required) - extracted from QuantizedCpuOps trait
            pub quantize_cpu: fn(&[f32]) -> crate::Result<Vec<u8>>,
            pub dequantize_cpu: fn(&[u8], &mut [f32]) -> crate::Result<()>,
            pub storage_size_in_bytes: fn(usize) -> usize,
            pub infer_element_count: fn(usize) -> usize,
            pub matmul_cpu: fn(&[f32], &[usize], &[u8], &[usize]) -> crate::Result<Vec<f32>>,

            // CUDA ops (optional) - extracted from QuantizedCudaOps trait
            #[cfg(feature = "cuda")]
            pub quantize_cuda: Option<fn(&cudarc::driver::CudaSlice<f32>) -> crate::Result<cudarc::driver::CudaSlice<u8>>>,
            #[cfg(feature = "cuda")]
            pub dequantize_cuda: Option<fn(&cudarc::driver::CudaSlice<u8>, &mut cudarc::driver::CudaSlice<f32>) -> crate::Result<()>>,
            #[cfg(feature = "cuda")]
            pub matmul_cuda: Option<fn(&cudarc::driver::CudaSlice<f32>, &[usize], &cudarc::driver::CudaSlice<u8>, &[usize]) -> crate::Result<cudarc::driver::CudaSlice<f32>>>,

            // Metal ops (optional) - extracted from QuantizedMetalOps trait
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

            #[inline]
            pub fn infer_element_count(id: QuantizedDType, data_len: usize) -> usize {
                match id {
                    #(QuantizedDType::#type_names => #type_names::default().infer_element_count(data_len),)*
                    QuantizedDType::External(name) => {
                        EXTERNAL_TYPE_REGISTRY.get()
                            .and_then(|r| r.read().ok())
                            .and_then(|m| m.get(name).map(|ops| (ops.infer_element_count)(data_len)))
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
                            .and_then(|m| m.get(name).map(|ops| (ops.storage_size_in_bytes)(num_elements)))
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
                        let ops = registry.read().unwrap().get(name).copied()
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        (ops.dequantize_cpu)(data, output)
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
                        let ops = registry.read().unwrap().get(name).copied()
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        (ops.quantize_cpu)(input)
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
                        let ops = registry.read().unwrap().get(name).copied()
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        (ops.matmul_cpu)(lhs_f32, lhs_shape, rhs_data, rhs_shape)
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

            // Level 0 (specialized): For types WITH CPU support - requires only 1 auto-deref
            // This impl is preferred over Level 1 when T implements QuantizedCpuOps
            // because method resolution favors impls requiring fewer auto-derefs
            #(
                impl MaybeCpuOps for &CpuWrap<#type_names>
                where
                    #type_names: candle_macros_types::QuantizedCpuOps,
                {
                    #[inline(always)]
                    fn try_dequantize_cpu(&self, data: &[u8], output: &mut [f32]) -> crate::Result<()> {
                        use candle_macros_types::QuantizedCpuOps;
                        <#type_names as Default>::default().dequantize(data, output).map_err(|e| crate::Error::Msg(e))
                    }

                    #[inline(always)]
                    fn try_quantize_cpu(&self, input: &[f32]) -> crate::Result<Vec<u8>> {
                        use candle_macros_types::QuantizedCpuOps;
                        <#type_names as Default>::default().quantize(input).map_err(|e| crate::Error::Msg(e))
                    }

                    #[inline(always)]
                    fn try_matmul_cpu(&self, lhs: &[f32], lhs_shape: &[usize], rhs: &[u8], rhs_shape: &[usize]) -> crate::Result<Vec<f32>> {
                        use candle_macros_types::QuantizedCpuOps;
                        <#type_names as Default>::default().matmul(lhs, lhs_shape, rhs, rhs_shape).map_err(|e| crate::Error::Msg(e))
                    }
                }
            )*

            // ============================================================================
            // CUDA Dispatch - Autoderef-Based Specialization (same pattern as CPU)
            // ============================================================================
            #[cfg(feature = "cuda")]
            struct CudaWrap<T>(std::marker::PhantomData<T>);

            #[cfg(feature = "cuda")]
            pub trait MaybeCudaOps {
                fn try_dequantize_cuda(&self, data: &cudarc::driver::CudaSlice<u8>, output: &mut cudarc::driver::CudaSlice<f32>) -> crate::Result<()>;
                fn try_quantize_cuda(&self, input: &cudarc::driver::CudaSlice<f32>) -> crate::Result<cudarc::driver::CudaSlice<u8>>;
                fn try_matmul_cuda(&self, lhs: &cudarc::driver::CudaSlice<f32>, lhs_shape: &[usize], rhs: &cudarc::driver::CudaSlice<u8>, rhs_shape: &[usize]) -> crate::Result<cudarc::driver::CudaSlice<f32>>;
            }

            // Level 1 (fallback): For types WITHOUT CUDA support
            #[cfg(feature = "cuda")]
            impl<T> MaybeCudaOps for CudaWrap<T>
            where
                T: candle_macros_types::QuantizedType,
            {
                #[inline(always)]
                fn try_dequantize_cuda(&self, _data: &cudarc::driver::CudaSlice<u8>, _output: &mut cudarc::driver::CudaSlice<f32>) -> crate::Result<()> {
                    Err(crate::Error::Msg(format!("Type '{}' does not implement CUDA dequantize", T::NAME)))
                }

                #[inline(always)]
                fn try_quantize_cuda(&self, _input: &cudarc::driver::CudaSlice<f32>) -> crate::Result<cudarc::driver::CudaSlice<u8>> {
                    Err(crate::Error::Msg(format!("Type '{}' does not implement CUDA quantize", T::NAME)))
                }

                #[inline(always)]
                fn try_matmul_cuda(&self, _lhs: &cudarc::driver::CudaSlice<f32>, _lhs_shape: &[usize], _rhs: &cudarc::driver::CudaSlice<u8>, _rhs_shape: &[usize]) -> crate::Result<cudarc::driver::CudaSlice<f32>> {
                    Err(crate::Error::Msg(format!("Type '{}' does not implement CUDA matmul", T::NAME)))
                }
            }

            // Level 0 (specialized): For types WITH CUDA support
            #(
                #[cfg(feature = "cuda")]
                impl MaybeCudaOps for &CudaWrap<#type_names>
                where
                    #type_names: candle_macros_types::QuantizedCudaOps,
                {
                    #[inline(always)]
                    fn try_dequantize_cuda(&self, data: &cudarc::driver::CudaSlice<u8>, output: &mut cudarc::driver::CudaSlice<f32>) -> crate::Result<()> {
                        use candle_macros_types::QuantizedCudaOps;
                        <#type_names as Default>::default().dequantize_cuda(data, output).map_err(|e| crate::Error::Msg(e))
                    }

                    #[inline(always)]
                    fn try_quantize_cuda(&self, input: &cudarc::driver::CudaSlice<f32>) -> crate::Result<cudarc::driver::CudaSlice<u8>> {
                        use candle_macros_types::QuantizedCudaOps;
                        <#type_names as Default>::default().quantize_cuda(input).map_err(|e| crate::Error::Msg(e))
                    }

                    #[inline(always)]
                    fn try_matmul_cuda(&self, lhs: &cudarc::driver::CudaSlice<f32>, lhs_shape: &[usize], rhs: &cudarc::driver::CudaSlice<u8>, rhs_shape: &[usize]) -> crate::Result<cudarc::driver::CudaSlice<f32>> {
                        use candle_macros_types::QuantizedCudaOps;
                        <#type_names as Default>::default().matmul_cuda(lhs, lhs_shape, rhs, rhs_shape).map_err(|e| crate::Error::Msg(e))
                    }
                }
            )*

            #[cfg(feature = "cuda")]
            #[inline]
            pub fn dequantize_cuda(
                id: QuantizedDType,
                data: &cudarc::driver::CudaSlice<u8>,
                output: &mut cudarc::driver::CudaSlice<f32>,
            ) -> crate::Result<()> {
                match id {
                    #(QuantizedDType::#type_names => (&&CudaWrap::<#type_names>(std::marker::PhantomData)).try_dequantize_cuda(data, output),)*
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let ops = registry.read().unwrap().get(name).copied()
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        ops.dequantize_cuda
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' does not support CUDA dequantize", name)))?
                            (data, output)
                    }
                }
            }

            #[cfg(feature = "cuda")]
            #[inline]
            pub fn quantize_cuda(
                id: QuantizedDType,
                input: &cudarc::driver::CudaSlice<f32>,
            ) -> crate::Result<cudarc::driver::CudaSlice<u8>> {
                match id {
                    #(QuantizedDType::#type_names => (&&CudaWrap::<#type_names>(std::marker::PhantomData)).try_quantize_cuda(input),)*
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let ops = registry.read().unwrap().get(name).copied()
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        ops.quantize_cuda
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' does not support CUDA quantize", name)))?
                            (input)
                    }
                }
            }

            #[cfg(feature = "cuda")]
            #[inline]
            pub fn matmul_cuda(
                id: QuantizedDType,
                lhs_data: &cudarc::driver::CudaSlice<f32>,
                lhs_shape: &[usize],
                rhs_id: QuantizedDType,
                rhs_data: &cudarc::driver::CudaSlice<u8>,
                rhs_shape: &[usize],
            ) -> crate::Result<cudarc::driver::CudaSlice<f32>> {
                match rhs_id {
                    #(QuantizedDType::#type_names => (&&CudaWrap::<#type_names>(std::marker::PhantomData)).try_matmul_cuda(lhs_data, lhs_shape, rhs_data, rhs_shape),)*
                    QuantizedDType::External(name) => {
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let ops = registry.read().unwrap().get(name).copied()
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        ops.matmul_cuda
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' does not support CUDA matmul", name)))?
                            (lhs_data, lhs_shape, rhs_data, rhs_shape)
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

            // Level 1 (fallback): For types WITHOUT Metal support
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

            // Level 0 (specialized): For types WITH Metal support
            #(
                #[cfg(feature = "metal")]
                impl MaybeMetalOps for &MetalWrap<#type_names>
                where
                    #type_names: candle_macros_types::QuantizedMetalOps,
                {
                    #[inline(always)]
                    fn try_dequantize_metal(&self, data: &metal::Buffer, output: &mut metal::Buffer) -> crate::Result<()> {
                        use candle_macros_types::QuantizedMetalOps;
                        <#type_names as Default>::default().dequantize_metal(data, output).map_err(|e| crate::Error::Msg(e))
                    }

                    #[inline(always)]
                    fn try_matmul_metal(&self, lhs: &metal::Buffer, lhs_shape: &[usize], rhs: &metal::Buffer, rhs_shape: &[usize]) -> crate::Result<metal::Buffer> {
                        use candle_macros_types::QuantizedMetalOps;
                        <#type_names as Default>::default().matmul_metal(lhs, lhs_shape, rhs, rhs_shape).map_err(|e| crate::Error::Msg(e))
                    }
                }
            )*

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
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let ops = registry.read().unwrap().get(name).copied()
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        ops.dequantize_metal
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' does not support Metal dequantize", name)))?
                            (data, output)
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
                        let registry = EXTERNAL_TYPE_REGISTRY.get()
                            .ok_or_else(|| crate::Error::Msg("External type registry not initialized".into()))?;
                        let ops = registry.read().unwrap().get(name).copied()
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' not registered", name)))?;
                        ops.matmul_metal
                            .ok_or_else(|| crate::Error::Msg(format!("External type '{}' does not support Metal matmul", name)))?
                            (lhs_data, lhs_shape, rhs_data, rhs_shape)
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
/// Type must implement QuantizedType and at least one of: QuantizedCpuOps, QuantizedCudaOps, or QuantizedMetalOps.
/// Other ops traits are optional - unimplemented operations will return errors when called.
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
            use candle_macros_types::QuantizedCpuOps;

            static DTYPE: OnceLock<QuantizedDType> = OnceLock::new();

            *DTYPE.get_or_init(|| {
                // Create a zero-sized instance for extracting trait methods
                let instance = #type_name::default();

                // Wrapper functions that convert from trait methods to function pointers
                fn quantize_wrapper(input: &[f32]) -> candle_core::Result<Vec<u8>> {
                    let instance = #type_name::default();
                    instance.quantize(input).map_err(|e| candle_core::Error::Msg(e))
                }

                fn dequantize_wrapper(data: &[u8], output: &mut [f32]) -> candle_core::Result<()> {
                    let instance = #type_name::default();
                    instance.dequantize(data, output).map_err(|e| candle_core::Error::Msg(e))
                }

                fn storage_size_wrapper(num_elements: usize) -> usize {
                    let instance = #type_name::default();
                    instance.storage_size_in_bytes(num_elements)
                }

                fn infer_count_wrapper(data_len: usize) -> usize {
                    let instance = #type_name::default();
                    instance.infer_element_count(data_len)
                }

                fn matmul_wrapper(
                    lhs: &[f32],
                    lhs_shape: &[usize],
                    rhs: &[u8],
                    rhs_shape: &[usize]
                ) -> candle_core::Result<Vec<f32>> {
                    let instance = #type_name::default();
                    instance.matmul(lhs, lhs_shape, rhs, rhs_shape)
                        .map_err(|e| candle_core::Error::Msg(e))
                }

                let ops = ExternalQuantOps {
                    size_in_bytes: <#type_name as candle_core::dtype::QuantizedType>::SIZE_IN_BYTES,
                    quantize_cpu: quantize_wrapper,
                    dequantize_cpu: dequantize_wrapper,
                    storage_size_in_bytes: storage_size_wrapper,
                    infer_element_count: infer_count_wrapper,
                    matmul_cpu: matmul_wrapper,

                    #[cfg(feature = "cuda")]
                    quantize_cuda: {
                        use candle_macros_types::QuantizedCudaOps;
                        fn wrapper(input: &cudarc::driver::CudaSlice<f32>) -> candle_core::Result<cudarc::driver::CudaSlice<u8>> {
                            let instance = #type_name::default();
                            instance.quantize_cuda(input).map_err(|e| candle_core::Error::Msg(e))
                        }
                        Some(wrapper)
                    },

                    #[cfg(feature = "cuda")]
                    dequantize_cuda: {
                        use candle_macros_types::QuantizedCudaOps;
                        fn wrapper(data: &cudarc::driver::CudaSlice<u8>, output: &mut cudarc::driver::CudaSlice<f32>) -> candle_core::Result<()> {
                            let instance = #type_name::default();
                            instance.dequantize_cuda(data, output).map_err(|e| candle_core::Error::Msg(e))
                        }
                        Some(wrapper)
                    },

                    #[cfg(feature = "cuda")]
                    matmul_cuda: {
                        use candle_macros_types::QuantizedCudaOps;
                        fn wrapper(
                            lhs: &cudarc::driver::CudaSlice<f32>,
                            lhs_shape: &[usize],
                            rhs: &cudarc::driver::CudaSlice<u8>,
                            rhs_shape: &[usize]
                        ) -> candle_core::Result<cudarc::driver::CudaSlice<f32>> {
                            let instance = #type_name::default();
                            instance.matmul_cuda(lhs, lhs_shape, rhs, rhs_shape)
                                .map_err(|e| candle_core::Error::Msg(e))
                        }
                        Some(wrapper)
                    },

                    #[cfg(feature = "metal")]
                    dequantize_metal: {
                        use candle_macros_types::QuantizedMetalOps;
                        fn wrapper(data: &metal::Buffer, output: &mut metal::Buffer) -> candle_core::Result<()> {
                            let instance = #type_name::default();
                            instance.dequantize_metal(data, output).map_err(|e| candle_core::Error::Msg(e))
                        }
                        Some(wrapper)
                    },

                    #[cfg(feature = "metal")]
                    matmul_metal: {
                        use candle_macros_types::QuantizedMetalOps;
                        fn wrapper(
                            lhs: &metal::Buffer,
                            lhs_shape: &[usize],
                            rhs: &metal::Buffer,
                            rhs_shape: &[usize]
                        ) -> candle_core::Result<metal::Buffer> {
                            let instance = #type_name::default();
                            instance.matmul_metal(lhs, lhs_shape, rhs, rhs_shape)
                                .map_err(|e| candle_core::Error::Msg(e))
                        }
                        Some(wrapper)
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
