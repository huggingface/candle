use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library, MTLDataType, MTLSize, NSUInteger,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::RwLock;

pub mod utils;
pub use utils::BufferOffset;
use utils::{get_block_dims, linear_split, EncoderParam, EncoderProvider};

const AFFINE: &str = include_str!("affine.metal");
const BINARY: &str = include_str!("binary.metal");
const CAST: &str = include_str!("cast.metal");
const CONV: &str = include_str!("conv.metal");
const FILL: &str = include_str!("fill.metal");
const INDEXING: &str = include_str!("indexing.metal");
// Current source: https://github.com/ivarflakstad/metal-flash-attention/tree/candle
#[cfg(not(target_os = "ios"))]
const MFA: &[u8] = include_bytes!("libMetalFlashAttention.metallib");
// Current source: https://github.com/philipturner/metal-flash-attention/releases/tag/v1.0.1
#[cfg(target_os = "ios")]
const MFA: &[u8] = include_bytes!("libMetalFlashAttention.ios.metallib");
const MLX_GEMM: &str = include_str!("mlx_gemm.metal");
const QUANTIZED: &str = include_str!("quantized.metal");
const RANDOM: &str = include_str!("random.metal");
const REDUCE: &str = include_str!("reduce.metal");
const SORT: &str = include_str!("sort.metal");
const TERNARY: &str = include_str!("ternary.metal");
const UNARY: &str = include_str!("unary.metal");
const SDPA: &str = include_str!("scaled_dot_product_attention.metal");
const MUL_AND_ACT: &str = include_str!("mul_and_act.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Gemm,
    Indexing,
    Mfa,
    Quantized,
    Random,
    Reduce,
    Sort,
    Ternary,
    Unary,
    Sdpa,
    MulAndAct,
}

pub mod copy2d {
    pub struct Kernel(pub &'static str);
    pub const FLOAT: Kernel = Kernel("copy2d_f32");
    pub const HALF: Kernel = Kernel("copy2d_f16");
    pub const BFLOAT: Kernel = Kernel("copy2d_bf16");
    pub const I64: Kernel = Kernel("copy2d_i64");
    pub const I32: Kernel = Kernel("copy2d_i32");
    pub const I16: Kernel = Kernel("copy2d_i16");
    pub const U32: Kernel = Kernel("copy2d_u32");
    pub const U8: Kernel = Kernel("copy2d_u8");
}

macro_rules! ops{
    ($($name:ident),+) => {

        pub mod contiguous {
        pub struct Kernel(pub &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_f16"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bf16"));
            pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64"));
            pub const I32: Kernel = Kernel(concat!(stringify!($name), "_i32"));
            pub const I16: Kernel = Kernel(concat!(stringify!($name), "_i16"));
            pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32"));
            pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8"));
        }
        )+
            pub mod copy {
                use super::Kernel;
                pub const FLOAT: Kernel = Kernel("copy_f32");
                pub const HALF: Kernel = Kernel("copy_f16");
                pub const BFLOAT: Kernel = Kernel("copy_bf16");
                pub const I64: Kernel = Kernel("copy_i64");
                pub const I32: Kernel = Kernel("copy_i32");
                pub const I16: Kernel = Kernel("copy_i16");
                pub const U32: Kernel = Kernel("copy_u32");
                pub const U8: Kernel = Kernel("copy_u8");
            }
        }

        pub mod contiguous_tiled {
        pub struct Kernel(pub &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32_tiled"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_f16_tiled"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bf16_tiled"));
            pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64_tiled"));
            pub const I32: Kernel = Kernel(concat!(stringify!($name), "_i32_tiled"));
            pub const I16: Kernel = Kernel(concat!(stringify!($name), "_i16_tiled"));
            pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32_tiled"));
            pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8_tiled"));
        }
        )+
            pub mod copy {
                use super::Kernel;
                pub const FLOAT: Kernel = Kernel("copy_f32_tiled");
                pub const HALF: Kernel = Kernel("copy_f16_tiled");
                pub const BFLOAT: Kernel = Kernel("copy_bf16_tiled");
                pub const I64: Kernel = Kernel("copy_i64_tiled");
                pub const I32: Kernel = Kernel("copy_i32_tiled");
                pub const I16: Kernel = Kernel("copy_i16_tiled");
                pub const U32: Kernel = Kernel("copy_u32_tiled");
                pub const U8: Kernel = Kernel("copy_u8_tiled");
            }
        }

        pub mod strided {
        pub struct Kernel(pub &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32_strided"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_f16_strided"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bf16_strided"));
            pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64_strided"));
            pub const I32: Kernel = Kernel(concat!(stringify!($name), "_i32_strided"));
            pub const I16: Kernel = Kernel(concat!(stringify!($name), "_i16_strided"));
            pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32_strided"));
            pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8_strided"));
        }
        )+
            pub mod copy {
                use super::Kernel;
                pub const FLOAT: Kernel = Kernel("copy_f32_strided");
                pub const HALF: Kernel = Kernel("copy_f16_strided");
                pub const BFLOAT: Kernel = Kernel("copy_bf16_strided");
                pub const I64: Kernel = Kernel("copy_i64_strided");
                pub const I32: Kernel = Kernel("copy_i32_strided");
                pub const I16: Kernel = Kernel("copy_i16_strided");
                pub const U32: Kernel = Kernel("copy_u32_strided");
                pub const U8: Kernel = Kernel("copy_u8_strided");
            }
        }
    };
}

pub mod unary {
    ops!(
        cos, sin, exp, sqr, sqrt, neg, log, gelu, abs, ceil, floor, relu, round, erf, gelu_erf,
        tanh, recip, silu, sign, sigmoid
    );
}
pub mod binary {
    ops!(add, sub, mul, div, min, max, eq, ne, le, lt, ge, gt);
}

#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    #[error("Could not lock kernel map: {0}")]
    LockError(String),
    #[error("Error while loading library: {0}")]
    LoadLibraryError(String),
    #[error("Error while loading function: {0:?}")]
    LoadFunctionError(String),
    #[error("Failed to create compute function")]
    FailedToCreateComputeFunction,
    #[error("Failed to create pipeline")]
    FailedToCreatePipeline(String),
    #[error("Invalid matmul arguments {lhs_stride:?} {rhs_stride:?} {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
    #[error("Sdpa {variation} head size was {got}, expectd {expected:?}")]
    SdpaHeadSizeMismatch {
        variation: &'static str,
        got: usize,
        expected: Vec<usize>,
    },
    #[error("Sdpa {variation} got dtype {got:?}")]
    SdpaHeadDTypeMismatch {
        variation: &'static str,
        got: SdpaDType,
    },
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

type Libraries = HashMap<Source, Library>;
type Pipelines = HashMap<(&'static str, Option<ConstantValues>), ComputePipelineState>;

#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(Libraries::new());
        let pipelines = RwLock::new(Pipelines::new());
        Self {
            libraries,
            pipelines,
        }
    }

    fn get_library_source(&self, source: Source) -> &'static str {
        match source {
            Source::Affine => AFFINE,
            Source::Binary => BINARY,
            Source::Cast => CAST,
            Source::Conv => CONV,
            Source::Fill => FILL,
            Source::Gemm => MLX_GEMM,
            Source::Indexing => INDEXING,
            Source::Quantized => QUANTIZED,
            Source::Random => RANDOM,
            Source::Reduce => REDUCE,
            Source::Sort => SORT,
            Source::Ternary => TERNARY,
            Source::Unary => UNARY,
            Source::Sdpa => SDPA,
            Source::MulAndAct => MUL_AND_ACT,
            Source::Mfa => panic!("Invalid lib"),
        }
    }

    /// Load the give library from its [`source`].
    /// If this has been previously loaded it will just fetch it from cache.
    pub fn load_library(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, MetalKernelError> {
        let mut libraries = self.libraries.write()?;
        if let Some(lib) = libraries.get(&source) {
            Ok(lib.clone())
        } else {
            let lib = match source {
                Source::Mfa => {
                    let source_data = MFA;
                    device.new_library_with_data(source_data).map_err(|e| {
                        MetalKernelError::LoadLibraryError(format!(
                            "Candle metal requires macosx > 13.0 or higher, cannot load mfa: {e}"
                        ))
                    })?
                }
                source => {
                    let source_content = self.get_library_source(source);
                    device
                        .new_library_with_source(source_content, &CompileOptions::new())
                        .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?
                }
            };
            libraries.insert(source, lib.clone());
            Ok(lib)
        }
    }

    fn load_function(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
        constants: Option<FunctionConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device, source)?
            .get_function(name, constants)
            .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
        Ok(func)
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source
    fn load_pipeline_with_constants(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let mut pipelines = self.pipelines.write()?;
        let key = (name, constants);
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let (name, constants) = key;
            let func = self.load_function(
                device,
                source,
                name,
                constants.as_ref().map(|c| c.function_constant_values()),
            )?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
            pipelines.insert((name, constants), pipeline.clone());

            Ok(pipeline)
        }
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source (without constants)
    pub fn load_pipeline(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        self.load_pipeline_with_constants(device, source, name, None)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_copy2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: copy2d::Kernel,
    input: &Buffer,
    output: &Buffer,
    d1: usize,
    d2: usize,
    src_s: usize,
    dst_s: usize,
    src_o_in_bytes: usize,
    dst_o_in_bytes: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, name.0)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            d1 as i64,
            d2 as i64,
            src_s as i64,
            dst_s as i64,
            (input, src_o_in_bytes),
            (output, dst_o_in_bytes)
        )
    );

    let grid_dims = MTLSize {
        width: d1 as u64,
        height: d2 as u64,
        depth: 1,
    };
    let group_dims = get_block_dims(d1 as u64, d2 as u64, 1);
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_unary_contiguous_tiled(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: unary::contiguous_tiled::Kernel,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel_name.0)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let tile_size = 2;
    let tiles = (length + tile_size - 1) / tile_size;

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, &input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_unary_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: unary::contiguous::Kernel,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel_name.0)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, &input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_unary_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: unary::strided::Kernel,
    shape: &[usize],
    input: BufferOffset,
    strides: &[usize],
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, name.0)?;

    let length: usize = shape.iter().product();
    let num_dims: usize = shape.len();
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(encoder, (length, num_dims, shape, strides, &input, &output));
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output.buffer, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_binary_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: binary::contiguous::Kernel,
    length: usize,
    left: BufferOffset,
    right: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Binary, kernel_name.0)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, &left, &right, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.use_resource(left.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(right.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_binary_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: binary::strided::Kernel,
    shape: &[usize],
    left_input: BufferOffset,
    left_strides: &[usize],
    right_input: BufferOffset,
    right_strides: &[usize],
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Binary, name.0)?;

    let num_dims: usize = shape.len();
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let width: usize = shape.iter().product();
    let length: usize = shape.iter().product();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, width);

    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            left_strides,
            right_strides,
            &left_input,
            &right_input,
            output
        )
    );
    encoder.use_resource(left_input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(right_input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_cast_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Cast, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, &input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_cast_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Cast, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();

    set_params!(
        encoder,
        (length, shape.len(), shape, input_strides, &input, output)
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let elements_to_sum = length / out_length;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, elements_to_sum, &input, output));

    let thread_group_count = MTLSize {
        width: out_length as u64,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (elements_to_sum as u64 + 2 - 1) / 2,
    )
    .next_power_of_two();

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let length: usize = shape.iter().product();
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let elements_to_sum = length / out_length;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (shape.len(), shape, strides, elements_to_sum, &input, output)
    );

    let thread_group_count = MTLSize {
        width: out_length as u64,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        elements_to_sum as u64,
    )
    .next_power_of_two();

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_last_softmax(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (input, input_offset),
            (output, output_offset)
        )
    );

    let out_length = length / elements_to_sum;

    let thread_group_count = MTLSize {
        width: out_length as u64,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        elements_to_sum as u64,
    )
    .next_power_of_two();

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

// Requires continuous input and mask
#[allow(clippy::too_many_arguments)]
pub fn call_last_attn_softmax(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input: &Buffer,
    input_offset: usize,
    mask: &Buffer,
    mask_offset: usize,
    input_shape: &[usize],
    mask_shape: &[usize],
    scale: f32,
    ty: SdpaDType,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = input_shape[input_shape.len() - 1] as i64;
    let ne01 = input_shape[input_shape.len() - 2] as i64;
    let ne02 = input_shape[input_shape.len() - 3] as i64;
    let ne03 = input_shape[input_shape.len() - 4] as i64;

    let elem_per_batch = if mask_shape.len() == 2 {
        0
    } else {
        let bs = input_shape[0];
        let el: usize = input_shape.iter().product();
        el / bs
    };

    let mut nth = 32; // SIMD width
    let name = if ne00 % 4 == 0 {
        while nth < ne00 / 4 && nth * ne01 * ne02 * ne03 < 256 {
            nth *= 2;
        }
        match ty {
            SdpaDType::F32 => "attn_soft_max_f32_4",
            SdpaDType::F16 => "attn_soft_max_f16_4",
            SdpaDType::BF16 => "attn_soft_max_bf16_4",
        }
    } else {
        while nth < ne00 && nth * ne01 * ne02 * ne03 < 256 {
            nth *= 2;
        }
        match ty {
            SdpaDType::F32 => "attn_soft_max_f32",
            SdpaDType::F16 => "attn_soft_max_f16",
            SdpaDType::BF16 => "attn_soft_max_bf16",
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (input, input_offset),
            (mask, mask_offset),
            (output, output_offset),
            ne00,
            ne01,
            ne02,
            elem_per_batch as i64,
            scale
        )
    );

    let thread_group_count = MTLSize {
        width: (ne01 * ne02 * ne03) as u64,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: nth as u64,
        height: 1,
        depth: 1,
    };

    encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (input, input_offset),
            output,
            (alpha, alpha_offset),
            eps
        )
    );

    let out_length = length / elements_to_sum;

    let thread_group_count = MTLSize {
        width: out_length as u64,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        elements_to_sum as u64,
    )
    .next_power_of_two();

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.set_threadgroup_memory_length(0, (width * 4).max(16) as u64);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_layer_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    beta: &Buffer,
    beta_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (input, input_offset),
            output,
            (alpha, alpha_offset),
            (beta, beta_offset),
            eps
        )
    );

    let out_length = length / elements_to_sum;

    let thread_group_count = MTLSize {
        width: out_length as u64,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        elements_to_sum as u64,
    )
    .next_power_of_two();

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.set_threadgroup_memory_length(0, (width * 8).max(32) as u64);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_i(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            bh,
            td,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.use_resource(src, metal::MTLResourceUsage::Read);
    encoder.use_resource(cos, metal::MTLResourceUsage::Read);
    encoder.use_resource(sin, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_thd(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    b: usize,
    t: usize,
    h: usize,
    d: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            b,
            t,
            h,
            d,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (b * t * h * d) / 2);
    encoder.use_resource(src, metal::MTLResourceUsage::Read);
    encoder.use_resource(cos, metal::MTLResourceUsage::Read);
    encoder.use_resource(sin, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    d: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            bh,
            td,
            d,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.use_resource(src, metal::MTLResourceUsage::Read);
    encoder.use_resource(cos, metal::MTLResourceUsage::Read);
    encoder.use_resource(sin, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_affine(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    size: usize,
    input: BufferOffset,
    output: &Buffer,
    mul: f32,
    add: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (size, mul, add, &input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_affine_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_stride: &[usize],
    output: &Buffer,
    mul: f32,
    add: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
    let size: usize = shape.iter().product();

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            size,
            shape.len(),
            shape,
            input_stride,
            mul,
            add,
            &input,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_powf(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    size: usize,
    input: BufferOffset,
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (size, mul, &input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_powf_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_stride: &[usize],
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
    let size: usize = shape.iter().product();

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (size, shape.len(), shape, input_stride, mul, &input, output)
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_elu(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    size: usize,
    input: BufferOffset,
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (size, mul, &input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_elu_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_stride: &[usize],
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
    let size: usize = shape.iter().product();

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (size, shape.len(), shape, input_stride, mul, &input, output)
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_where_cond_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    cond: BufferOffset,
    cond_stride: &[usize],
    left: BufferOffset,
    left_stride: &[usize],
    right: BufferOffset,
    right_stride: &[usize],
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Ternary, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let size: usize = shape.iter().product();
    let rank = shape.len();

    set_params!(
        encoder,
        (
            size,
            rank,
            shape,
            cond_stride,
            left_stride,
            right_stride,
            &cond,
            &left,
            &right,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);

    encoder.use_resource(cond.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(left.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(right.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_index_select(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    ids_size: usize,
    dim: usize,
    contiguous: bool,
    src_dims: &[usize],
    src_strides: &[usize],
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let src_dim_size = shape[dim];
    let dst_el = ids_size * left_size * right_size;

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            ids_size,
            contiguous,
            src_dims,
            src_strides,
            &input,
            &ids,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(ids.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_gather(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    ids_size: usize,
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let src_dim_size = shape[dim];
    let dst_el = ids_size * left_size * right_size;

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            ids_size,
            &input,
            &ids,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(ids.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_scatter_add(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    src_shape: &[usize],
    dst_shape: &[usize],
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = src_shape[..dim].iter().product();
    let right_size: usize = src_shape[dim + 1..].iter().product();
    let src_dim_size = src_shape[dim];
    let dst_el = left_size * right_size;
    let dst_dim_size = dst_shape[dim];

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            dst_dim_size,
            &input,
            &ids,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(ids.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_index_add(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    src_shape: &[usize],
    dst_shape: &[usize],
    ids_shape: &[usize],
    dim: usize,
    input: BufferOffset,
    ids: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = src_shape[..dim].iter().product();
    let right_size: usize = src_shape[dim + 1..].iter().product();
    let src_dim_size = src_shape[dim];
    let dst_el = left_size * right_size;
    let dst_dim_size = dst_shape[dim];
    let ids_dim_size = ids_shape[0];

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            dst_dim_size,
            ids_dim_size,
            &input,
            &ids,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(ids.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[derive(Debug, PartialEq)]
pub enum Value {
    USize(usize),
    Bool(bool),
    F32(f32),
    U16(u16),
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::F32(v) => v.to_bits().hash(state),
            Value::USize(v) => v.hash(state),
            Value::U16(v) => v.hash(state),
            Value::Bool(v) => v.hash(state),
        }
    }
}

impl Value {
    fn data_type(&self) -> MTLDataType {
        match self {
            Value::USize(_) => MTLDataType::UInt,
            Value::F32(_) => MTLDataType::Float,
            Value::U16(_) => MTLDataType::UShort,
            Value::Bool(_) => MTLDataType::Bool,
        }
    }
}

/// Not true, good enough for our purposes.
impl Eq for Value {}

#[derive(Debug, Eq, PartialEq, Hash)]
struct ConstantValues(Vec<(usize, Value)>);

impl ConstantValues {
    pub fn new(values: Vec<(usize, Value)>) -> Self {
        Self(values)
    }

    fn function_constant_values(&self) -> FunctionConstantValues {
        let f = FunctionConstantValues::new();
        for (index, value) in &self.0 {
            let ty = value.data_type();
            match value {
                Value::USize(v) => {
                    f.set_constant_value_at_index(
                        v as *const usize as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::F32(v) => {
                    f.set_constant_value_at_index(
                        v as *const f32 as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::U16(v) => {
                    f.set_constant_value_at_index(
                        v as *const u16 as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::Bool(v) => {
                    f.set_constant_value_at_index(
                        v as *const bool as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
            }
        }
        f
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_gemm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
    alpha: f32,
    beta: f32,
) -> Result<(), MetalKernelError> {
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // lhs has shape b, m, k
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let a_trans = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        false
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        true
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?;
    };
    // rhs has shape b, k, n
    let b_trans = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        false
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        true
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?;
    };
    let d_trans = false;
    let batched = b > 1;
    let fused_activation = false;
    let fused_bias = false;
    let (m_simd, n_simd, k_simd, m_splits, n_splits) = if m == 1 {
        let m_simd = 8;
        let n_simd = 8;
        let k_simd = 64;
        let m_splits = 1;
        let n_splits = 1;
        (m_simd, n_simd, k_simd, m_splits, n_splits)
    } else {
        let m_simd = 40;
        let n_simd = 40;
        let k_simd = 32;
        let m_splits = 1;
        let n_splits = 1;
        (m_simd, n_simd, k_simd, m_splits, n_splits)
    };
    let constants = Some(ConstantValues::new(vec![
        (0, Value::USize(m)),
        (1, Value::USize(n)),
        (2, Value::USize(k)),
        (10, Value::Bool(a_trans)),
        (11, Value::Bool(b_trans)),
        (13, Value::Bool(d_trans)),
        (20, Value::F32(alpha)),
        (21, Value::F32(beta)),
        (100, Value::Bool(batched)),
        (101, Value::Bool(fused_activation)),
        // Garbage
        (102, Value::Bool(false)),
        (103, Value::Bool(false)),
        (113, Value::Bool(false)),
        (50_000, Value::Bool(false)),
        // End garbage
        (200, Value::U16(m_simd)),
        (201, Value::U16(n_simd)),
        (202, Value::U16(k_simd)),
        (210, Value::U16(m_splits)),
        (211, Value::U16(n_splits)),
        (50_001, Value::Bool(fused_bias)),
    ]));
    let pipeline = kernels.load_pipeline_with_constants(device, Source::Mfa, name, constants)?;
    let m_group = m_simd * m_splits;
    let n_group = n_simd * n_splits;

    let a_block_length = m_group * k_simd;
    let b_block_length = k_simd * n_group;

    let mut block_elements = a_block_length + b_block_length;
    if (m % 8 != 0) && (n % 8 != 0) {
        let c_block_length = m_group * n_group;
        block_elements = std::cmp::max(c_block_length, block_elements)
    }
    if fused_bias {
        if d_trans {
            block_elements = std::cmp::max(block_elements, m_group);
        } else {
            block_elements = std::cmp::max(block_elements, n_group);
        }
    }
    let bytes = match name {
        "sgemm" => 4,
        "hgemm" => 2,
        "bgemm" => 2,
        other => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "{other} is not a valid kernel for gemm"
            )));
        }
    };
    let block_bytes = block_elements * bytes;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_threadgroup_memory_length(0, block_bytes.into());
    encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as NSUInteger);
    encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as NSUInteger);
    encoder.set_buffer(2, Some(output), 0);
    // TODO Tensor D

    let grid_z = b;
    if batched {
        let byte_stride_a: usize = lhs_stride[lhs_stride.len() - 3] * bytes as usize;
        let byte_stride_b: usize = rhs_stride[rhs_stride.len() - 3] * bytes as usize;
        let byte_stride_c = m * n * bytes as usize;
        // TODO byte_stride_d
        let byte_stride_d = 0;

        let buffer: Vec<u64> = vec![
            byte_stride_a as _,
            byte_stride_b as _,
            byte_stride_c as _,
            byte_stride_d as _,
        ];
        encoder.set_bytes(
            10,
            (buffer.len() * core::mem::size_of::<u64>()) as NSUInteger,
            buffer.as_ptr() as *const NSUInteger as *const c_void,
        );
    }

    let grid_size = MTLSize {
        width: divide(n, n_group.into()),
        height: divide(m, m_group.into()),
        depth: grid_z as NSUInteger,
    };
    let group_size = MTLSize {
        width: 32 * (m_splits as u64) * (n_splits as u64),
        height: 1,
        depth: 1,
    };
    encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum SdpaDType {
    BF16,
    F16,
    F32,
}

/// SDPA full is supported when:
/// - q head dim == 64, 128
/// - no mask
/// - q heads == kv heads
/// - final type != bf16 (TODO maybe just template this kernel too?)
/// - q,k,v are contiguous
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_full(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_buffer: &Buffer,
    v_offset: usize,
    v_buffer: &Buffer,
    output: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct MLXFastAttentionParams {
        m: i32,
        n: i32,
        k: i32,

        ldq: i32, // ldq == ldo
        ldk: i32,
        ldv: i32,
        lds: i32,
        ldo: i32,

        tiles_n: i32,
        tiles_m: i32,

        batch_stride_q: i32,
        batch_stride_k: i32,
        batch_stride_v: i32,
        batch_stride_o: i32,

        swizzle_log: i32,
        gemm_n_iterations_aligned: i32,
        gemm_k_iterations_aligned: i32,
        gemm_sv_m_block_iterations: i32,

        batch_ndim: i32,
        alpha: f32,
        softcapping: f32,
    }

    let bk = q_shape.last().unwrap();

    const BN: usize = 16;
    const BM: usize = 16;
    const WM: usize = 2;
    const WN: usize = 2;

    let name = match (bk, itype) {
        (32, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_32_itype_half",
        (64, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_64_itype_half",
        (96, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_96_itype_half",
        (128, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_128_itype_half",
        (256, SdpaDType::F16) => "steel_gemm_attention_bm_16_bn_16_bk_256_itype_half",
        (32, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_32_itype_float",
        (64, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_64_itype_float",
        (96, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_96_itype_float",
        (128, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_128_itype_float",
        (256, SdpaDType::F32) => "steel_gemm_attention_bm_16_bn_16_bk_256_itype_float",
        (other, SdpaDType::F16 | SdpaDType::F32) => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "full",
                got: *other,
                expected: vec![32, 64, 96, 128, 256],
            })
        }
        (_, SdpaDType::BF16) => {
            return Err(MetalKernelError::SdpaHeadDTypeMismatch {
                variation: "full",
                got: SdpaDType::BF16,
            })
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Sdpa, &name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // q = (bs, qhead, seq, hidden)
    // k/v = (bs, kv_head, seq, hidden)

    let qseq = q_shape[q_shape.len() - 2];

    let m = q_shape[q_shape.len() - 2];
    let n = m;
    let k = q_shape[q_shape.len() - 1];
    let bs_out = q_shape[0] * q_shape[1];

    let batch_shape = [q_shape[0] * q_shape[1]];
    let dk = q_shape[q_shape.len() - 1];
    let ldq = dk;
    let ldk = dk;
    let ldv = dk;
    let lds = BN;
    let ldo = dk;

    let tn = 1;
    let tm = (m + BM - 1) / BM;

    let b_stride_q = dk * qseq;
    let b_stride_k = dk * qseq;
    let b_stride_v = dk * qseq;
    let b_stride_o = dk * qseq;
    let swizzle_log = 0;
    let gemm_n_iterations_aligned = (n + BN - 1) / BN;
    let gemm_k_iterations_aligned = (k + bk - 1) / bk;
    let gemm_sv_m_block_iterations = (m + BM - 1) / BM;
    let batch_ndim = batch_shape.len();

    let alpha = if softcapping != 1. {
        alpha / softcapping
    } else {
        alpha
    };

    let params = MLXFastAttentionParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        ldq: ldq as i32,
        ldk: ldk as i32,
        ldv: ldv as i32,
        lds: lds as i32,
        ldo: ldo as i32,
        tiles_n: tn,
        tiles_m: tm as i32,
        batch_stride_q: b_stride_q as i32,
        batch_stride_k: b_stride_k as i32,
        batch_stride_v: b_stride_v as i32,
        batch_stride_o: b_stride_o as i32,
        swizzle_log,
        gemm_n_iterations_aligned: gemm_n_iterations_aligned as i32,
        gemm_k_iterations_aligned: gemm_k_iterations_aligned as i32,
        gemm_sv_m_block_iterations: gemm_sv_m_block_iterations as i32,
        batch_ndim: batch_ndim as i32,
        alpha,
        softcapping,
    };
    let batch_strides = [b_stride_q, b_stride_k, b_stride_v, b_stride_o];

    impl EncoderParam for MLXFastAttentionParams {
        fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
            encoder.set_bytes(
                position,
                core::mem::size_of::<MLXFastAttentionParams>() as u64,
                &data as *const MLXFastAttentionParams as *const c_void,
            );
        }
    }

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            output,
            params,
            &batch_shape[..],
            &batch_strides[..]
        )
    );

    let grid_dims = MTLSize {
        width: 1,
        height: tm as u64,
        depth: bs_out as u64,
    };
    let group_dims = MTLSize {
        width: 32,
        height: WM as u64,
        depth: WN as u64,
    };
    encoder.use_resource(q_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(k_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(v_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// SDPA vector is supported when:
/// - q head dim == 64, 96, 128
/// - no mask
/// - q,k,v are contiguous
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let bk = q_shape.last().unwrap();

    let gqa_factor = (q_shape[1] / k_shape[1]) as i32;
    let n = k_shape[2] as i32;
    let b = (q_shape[0] * q_shape[1]) as i32;
    let kstride = k_stride[1];
    let vstride = v_stride[1];

    let name = match (bk, itype) {
        (32, SdpaDType::F16) => "sdpa_vector_float16_t_32",
        (64, SdpaDType::F16) => "sdpa_vector_float16_t_64",
        (96, SdpaDType::F16) => "sdpa_vector_float16_t_96",
        (128, SdpaDType::F16) => "sdpa_vector_float16_t_128",
        (256, SdpaDType::F16) => "sdpa_vector_float16_t_256",
        (32, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_32",
        (64, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_64",
        (96, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_96",
        (128, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_128",
        (256, SdpaDType::BF16) => "sdpa_vector_bfloat16_t_256",
        (32, SdpaDType::F32) => "sdpa_vector_float_32",
        (64, SdpaDType::F32) => "sdpa_vector_float_64",
        (96, SdpaDType::F32) => "sdpa_vector_float_96",
        (128, SdpaDType::F32) => "sdpa_vector_float_128",
        (256, SdpaDType::F32) => "sdpa_vector_float_256",
        (other, _) => {
            return Err(MetalKernelError::SdpaHeadSizeMismatch {
                variation: "vector",
                got: *other,
                expected: vec![32, 64, 96, 128, 256],
            })
        }
    };

    let alpha = if softcapping != 1. {
        alpha / softcapping
    } else {
        alpha
    };

    let constants = Some(ConstantValues::new(vec![(
        20,
        Value::Bool(/* sdpa_vector_has_mask */ false),
    )]));

    let pipeline = kernels.load_pipeline_with_constants(device, Source::Sdpa, &name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // q = (bs, qhead, seq, hidden)
    // k/v = (bs, kv_head, kv_seq, hidden)

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            output,
            gqa_factor,
            n,
            kstride,
            vstride,
            alpha,
            softcapping
        )
    );

    let grid_dims = MTLSize {
        width: 1,
        height: b as u64,
        depth: 1 as u64,
    };
    let group_dims = MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(q_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(k_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(v_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

pub const SDPA_2PASS_BLOCKS: usize = 32;

/// SDPA vector 2pass is supported when:
/// - q head dim == 64, 96, 128
/// - no mask
/// - q,k,v are contiguous
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_2pass(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_stride: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_stride: &[usize],
    v_buffer: &Buffer,
    output: &Buffer,
    intermediate: &Buffer,
    sums: &Buffer,
    maxs: &Buffer,
    alpha: f32,
    softcapping: f32,
    itype: SdpaDType,
) -> Result<(), MetalKernelError> {
    let bk = q_shape.last().unwrap();

    // First pass
    {
        let name_pass1 = match (bk, itype) {
            (32, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_1_float16_t_256",
            (32, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_32",
            (64, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_64",
            (96, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_96",
            (128, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_128",
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_1_bfloat16_t_256",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_1_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_1_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_1_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_1_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_1_float_256",
            (other, _) => {
                return Err(MetalKernelError::SdpaHeadSizeMismatch {
                    variation: "vector_2pass_1",
                    got: *other,
                    expected: vec![32, 64, 96, 128, 256],
                })
            }
        };

        let gqa_factor = (q_shape[1] / k_shape[1]) as i32;
        let n = k_shape[2] as i32;
        let b = (q_shape[0] * q_shape[1]) as i32;
        let kstride = k_stride[1];
        let vstride = v_stride[1];

        let alpha = if softcapping != 1. {
            alpha / softcapping
        } else {
            alpha
        };

        let constants = Some(ConstantValues::new(vec![(
            20,
            Value::Bool(/* sdpa_vector_has_mask */ false),
        )]));

        let pipeline =
            kernels.load_pipeline_with_constants(device, Source::Sdpa, &name_pass1, constants)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        // q = (bs, qhead, seq, hidden)
        // k/v = (bs, kv_head, kv_seq, hidden)

        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                intermediate,
                sums,
                maxs,
                gqa_factor,
                n,
                kstride,
                vstride,
                alpha,
                softcapping
            )
        );

        let grid_dims = MTLSize {
            width: 1,
            height: b as u64,
            depth: SDPA_2PASS_BLOCKS as u64,
        };
        let group_dims = MTLSize {
            width: 8 * 32,
            height: 1,
            depth: 1,
        };
        encoder.use_resource(q_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(k_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(v_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(intermediate, metal::MTLResourceUsage::Write);
        encoder.use_resource(sums, metal::MTLResourceUsage::Write);
        encoder.use_resource(maxs, metal::MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    // Final pass
    {
        let name_pass2 = match (bk, itype) {
            (32, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_32",
            (64, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_64",
            (96, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_96",
            (128, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_128",
            (256, SdpaDType::F16) => "sdpa_vector_2pass_2_float16_t_256",
            (32, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_32",
            (64, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_64",
            (96, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_96",
            (128, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_128",
            (256, SdpaDType::BF16) => "sdpa_vector_2pass_2_bfloat16_t_256",
            (32, SdpaDType::F32) => "sdpa_vector_2pass_2_float_32",
            (64, SdpaDType::F32) => "sdpa_vector_2pass_2_float_64",
            (96, SdpaDType::F32) => "sdpa_vector_2pass_2_float_96",
            (128, SdpaDType::F32) => "sdpa_vector_2pass_2_float_128",
            (256, SdpaDType::F32) => "sdpa_vector_2pass_2_float_256",
            (other, _) => {
                return Err(MetalKernelError::SdpaHeadSizeMismatch {
                    variation: "vector_2pass_2",
                    got: *other,
                    expected: vec![32, 64, 96, 128, 256],
                })
            }
        };

        let b = (q_shape[0] * q_shape[1]) as i32;

        let pipeline = kernels.load_pipeline(device, Source::Sdpa, &name_pass2)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        // q = (bs, qhead, seq, hidden)
        // k/v = (bs, kv_head, kv_seq, hidden)

        set_params!(encoder, (intermediate, sums, maxs, output));

        let grid_dims = MTLSize {
            width: 1,
            height: b as u64,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        encoder.use_resource(intermediate, metal::MTLResourceUsage::Write);
        encoder.use_resource(sums, metal::MTLResourceUsage::Write);
        encoder.use_resource(maxs, metal::MTLResourceUsage::Write);
        encoder.use_resource(output, metal::MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_im2col1d_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (k_size, stride, padding, dilation): (usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let l_out = (shape[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
    let dst_el = shape[0] * l_out * shape[1] * k_size;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (dst_el, l_out, k_size, stride, padding, dilation, shape, strides, &input, output)
    );
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_col2im1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    k_size: usize,
    stride: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let l_in = shape[1];
    let c_out = shape[2];
    let l_out = (l_in - 1) * stride + k_size;
    let dst_el = shape[0] * c_out * l_out;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (dst_el, l_out, l_in, c_out, k_size, stride, &input, output)
    );
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_im2col_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (h_k, w_k, stride, padding, dilation): (usize, usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;

    let h = shape[2];
    let w = shape[3];
    let h_out = (h + 2 * padding - dilation * (h_k - 1) - 1) / stride + 1;
    let w_out = (w + 2 * padding - dilation * (w_k - 1) - 1) / stride + 1;

    let dst_el = shape[0] * h_out * w_out * shape[1] * h_k * w_k;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            dst_el, h_out, w_out, h_k, w_k, stride, padding, dilation, shape, strides, &input,
            output
        )
    );
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_upsample_nearest_2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let scale_w = shape[2] as f32 / out_w as f32;
    let scale_h = shape[3] as f32 / out_h as f32;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (out_w, out_h, scale_w, scale_h, shape, strides, &input, output)
    );
    encoder.use_resource(input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_random_uniform(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    min: f32,
    max: f32,
    length: usize,
    seed: &Buffer,
    buffer: &Buffer,
) -> Result<(), MetalKernelError> {
    if min >= max {
        return Err(MetalKernelError::LoadLibraryError(
            "min must be less than max".to_string(),
        ));
    }
    let pipeline = kernels.load_pipeline(device, Source::Random, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();

    let odd = (length % 2 != 0) as usize;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length / 2 + odd);

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, min, max, seed, buffer));

    encoder.use_resource(
        seed,
        metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write,
    );
    encoder.use_resource(buffer, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_random_normal(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    mean: f32,
    stddev: f32,
    length: usize,
    seed: &Buffer,
    buffer: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Random, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();

    let odd = (length % 2 != 0) as usize;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length / 2 + odd);

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, mean, stddev, seed, buffer));

    encoder.use_resource(
        seed,
        metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write,
    );
    encoder.use_resource(buffer, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub enum GgmlDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    F16,
    F32,
    BF16,
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    let (nth0, nth1, align) = match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1 => {
            let nth0 = 8;
            let nth1 = 8;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::Q2K => {
            // Fixing a bug in Metal for GGML
            // https://github.com/ggerganov/llama.cpp/blob/b8109bc0139f15a5b321909f47510b89dca47ffc/ggml-metal.m#L1576
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q4K => {
            let nth0 = 4;
            let nth1 = 8;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q3K | GgmlDType::Q5K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q6K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 2;
            (nth0, nth1, align)
        }
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K => {
            // Original implem uses rows
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::F32 => {
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
    };
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as u64,
        depth: (ne12 * ne13) as u64,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mv_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f32",
        GgmlDType::Q8_1 => "kernel_mul_mv_q8_1_f32",
        GgmlDType::Q2K => "kernel_mul_mv_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_q6_K_f32",
        GgmlDType::Q8K => "kernel_mul_mv_q8_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mv_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mv_f32_f32",
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            (dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(lhs, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs, metal::MTLResourceUsage::Read);
    encoder.use_resource(dst, metal::MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// - src0 is usually weight
/// - src1 is usually xs
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = src0_shape[src0_shape.len() - 1] as i64;
    let ne01 = src0_shape[src0_shape.len() - 2] as i64;
    let ne02 = src0_shape[src0_shape.len() - 3] as i64;
    let ne03 = src0_shape[src0_shape.len() - 4] as i64;

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;
    let nb03 = src0_stride[src0_stride.len() - 4] as i64;

    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = src1_shape[src1_shape.len() - 4] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let nb13 = src1_stride[src1_stride.len() - 4] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64;
    let ne1 = dst_shape[dst_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    let thread_groups_count = MTLSize {
        width: divide(ne11 as usize, 32),
        height: divide(ne01 as usize, 64),
        depth: (ne12 * ne13) as u64,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mm_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mm_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mm_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mm_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f32",
        GgmlDType::Q8_1 => "kernel_mul_mm_q8_1_f32",
        GgmlDType::Q2K => "kernel_mul_mm_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mm_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32",
        GgmlDType::Q8K => "kernel_mul_mm_q8_K_f32",
        GgmlDType::F16 => "kernel_mul_mm_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mm_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mm_f32_f32",
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            src0,
            (src1, src1_offset),
            (dst, dst_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0, metal::MTLResourceUsage::Read);
    encoder.use_resource(src1, metal::MTLResourceUsage::Read);
    encoder.use_resource(dst, metal::MTLResourceUsage::Write);

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

fn divide(m: usize, b: usize) -> NSUInteger {
    ((m + b - 1) / b) as NSUInteger
}

#[allow(clippy::too_many_arguments)]
pub fn call_pool2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    w_k: usize,
    h_k: usize,
    w_stride: usize,
    h_stride: usize,
    input: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let pipeline: ComputePipelineState = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (w_k, h_k, w_stride, h_stride, shape, strides, input, output)
    );
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dilation: usize,
    stride: usize,
    padding: usize,
    out_padding: usize,
    c_out: usize,
    l_out: usize,
    b_size: usize,
    src_shape: &[usize],
    src_strides: &[usize],
    kernel_shape: &[usize],
    kernel_strides: &[usize],
    input: &Buffer,
    input_offset: usize,
    kernel: &Buffer,
    kernel_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = c_out * l_out * b_size;
    let pipeline: ComputePipelineState = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            l_out,
            stride,
            padding,
            out_padding,
            dilation,
            src_shape,
            src_strides,
            kernel_shape,
            kernel_strides,
            (input, input_offset),
            (kernel, kernel_offset),
            output
        )
    );
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(kernel, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

pub struct CallConvTranspose2dCfg<'a> {
    pub dilation: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub c_out: usize,
    pub out_w: usize,
    pub out_h: usize,
    pub b_size: usize,
    pub input_dims: &'a [usize],
    pub input_stride: &'a [usize],
    pub kernel_dims: &'a [usize],
    pub kernel_stride: &'a [usize],
    pub input_offset: usize,
    pub kernel_offset: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    cfg: CallConvTranspose2dCfg,
    input: &Buffer,
    kernel: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_el = cfg.c_out * cfg.out_w * cfg.out_h * cfg.b_size;
    let pipeline: ComputePipelineState = kernels.load_pipeline(device, Source::Conv, name)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            cfg.out_w,
            cfg.out_h,
            cfg.stride,
            cfg.padding,
            cfg.output_padding,
            cfg.dilation,
            cfg.input_dims,
            cfg.input_stride,
            cfg.kernel_dims,
            cfg.kernel_stride,
            (input, cfg.input_offset),
            (kernel, cfg.kernel_offset),
            output
        )
    );
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(kernel, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_arg_sort(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    nrows: usize,
    ncols: usize,
    ncols_pad: usize,
    src: BufferOffset,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Sort, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (&src, dst, ncols as i64, ncols_pad as i64));

    let thread_group_count = MTLSize {
        width: 1,
        height: nrows as u64,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: ncols_pad as u64,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(src.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(dst, metal::MTLResourceUsage::Write);
    encoder.set_threadgroup_memory_length(0, (ncols_pad * 4).max(16) as u64);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum GemmDType {
    BF16,
    F16,
    F32,
}

#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct GemmParams {
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldd: i32,
        tiles_n: i32,
        tiles_m: i32,
        batch_stride_a: isize,
        batch_stride_b: isize,
        batch_stride_d: isize,
        swizzle_log: i32,
        gemm_k_iterations_aligned: i32,
        batch_ndim: i32,
    }
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // lhs has shape b, m, k
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?;
    };
    // rhs has shape b, k, n
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?;
    };
    let (bm, bn, bk, wn, wm) = (32, 32, 16, 2, 2);
    // https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/matmul.cpp#L422
    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(/* has_batch */ b > 1)),
        (100, Value::Bool(/* use_out_source */ false)),
        (110, Value::Bool(/* do_axpby */ false)),
        (200, Value::Bool(/* align_m */ m % bm == 0)),
        (201, Value::Bool(/* align_n */ n % bn == 0)),
        (202, Value::Bool(/* align_k */ k % bk == 0)),
        (300, Value::Bool(/* do_gather */ false)),
    ]));

    let swizzle_log = 0;
    let tile = 1 << swizzle_log;
    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let tn = tn * tile;
    let tm = tm.div_ceil(tile);

    let batch_stride_a = if lhs_stride.len() > 2 {
        lhs_stride[lhs_stride.len() - 3]
    } else {
        m * k
    };
    let batch_stride_b = if rhs_stride.len() > 2 {
        rhs_stride[rhs_stride.len() - 3]
    } else {
        n * k
    };

    let gemm_params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda,
        ldb,
        ldd: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
        batch_stride_a: batch_stride_a as isize,
        batch_stride_b: batch_stride_b as isize,
        batch_stride_d: (m * n) as isize,
        batch_ndim: 1i32,
        gemm_k_iterations_aligned: (k / bk) as i32,
    };
    let batch_strides = [gemm_params.batch_stride_a, gemm_params.batch_stride_b];

    // TODO(laurent): generate the name
    // template [[host_name("gemm_" #tname "_"  #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn)]]
    let name = match (dtype, a_trans, b_trans) {
        (GemmDType::F32, false, false) => "gemm_nn_f32_f32_32_32_16_2_2",
        (GemmDType::F32, true, false) => "gemm_tn_f32_f32_32_32_16_2_2",
        (GemmDType::F32, false, true) => "gemm_nt_f32_f32_32_32_16_2_2",
        (GemmDType::F32, true, true) => "gemm_tt_f32_f32_32_32_16_2_2",
        (GemmDType::BF16, false, false) => "gemm_nn_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, true, false) => "gemm_tn_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, false, true) => "gemm_nt_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, true, true) => "gemm_tt_bf16_bf16_32_32_16_2_2",
        (GemmDType::F16, false, false) => "gemm_nn_f16_f16_32_32_16_2_2",
        (GemmDType::F16, true, false) => "gemm_tn_f16_f16_32_32_16_2_2",
        (GemmDType::F16, false, true) => "gemm_nt_f16_f16_32_32_16_2_2",
        (GemmDType::F16, true, true) => "gemm_tt_f16_f16_32_32_16_2_2",
    };
    let pipeline = kernels.load_pipeline_with_constants(device, Source::Gemm, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as NSUInteger);
    encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as NSUInteger);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_bytes(
        4,
        std::mem::size_of::<GemmParams>() as u64,
        &gemm_params as *const GemmParams as *const c_void,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<GemmParams>() as u64,
        &gemm_params as *const GemmParams as *const c_void,
    );
    encoder.set_bytes(
        6, // batch_shape
        std::mem::size_of::<i32>() as u64,
        &(b as i32) as *const i32 as *const c_void,
    );
    encoder.set_bytes(
        7,
        (std::mem::size_of::<isize>() * batch_strides.len()) as u64,
        batch_strides.as_ptr() as *const c_void,
    );

    let grid_size = MTLSize {
        width: tn as u64,
        height: tm as u64,
        depth: /* batch_size_out */ b as u64,
    };
    let group_size = MTLSize {
        width: 32,
        height: wn,
        depth: wm,
    };
    encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_mlx_addmm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    c_stride: &[usize],
    c_offset: usize,
    c_buffer: &Buffer,
    output: &Buffer,
    alpha: f32,
    beta: f32,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct GemmParams {
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldd: i32,
        tiles_n: i32,
        tiles_m: i32,
        batch_stride_a: isize,
        batch_stride_b: isize,
        batch_stride_d: isize,
        swizzle_log: i32,
        gemm_k_iterations_aligned: i32,
        batch_ndim: i32,
    }

    #[derive(Debug)]
    #[repr(C)]
    struct GEMMAddMMParams {
        ldc: i32,
        fdc: i32,
        batch_stride_c: isize,
        alpha: f32,
        beta: f32,
    }

    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // lhs has shape b, m, k
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?;
    };
    // rhs has shape b, k, n
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?;
    };
    let (bm, bn, bk, wn, wm) = (32, 32, 16, 2, 2);
    // https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/matmul.cpp#L422
    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(/* has_batch */ b > 1)),
        (100, Value::Bool(/* use_out_source */ true)),
        (110, Value::Bool(/* do_axpby */ true)),
        (200, Value::Bool(/* align_m */ m % bm == 0)),
        (201, Value::Bool(/* align_n */ n % bn == 0)),
        (202, Value::Bool(/* align_k */ k % bk == 0)),
        (300, Value::Bool(/* do_gather */ false)),
    ]));

    let swizzle_log = 0;
    let tile = 1 << swizzle_log;
    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let tn = tn * tile;
    let tm = tm.div_ceil(tile);

    let batch_stride_a = if lhs_stride.len() > 2 {
        lhs_stride[lhs_stride.len() - 3]
    } else {
        m * k
    };
    let batch_stride_b = if rhs_stride.len() > 2 {
        rhs_stride[rhs_stride.len() - 3]
    } else {
        n * k
    };

    let gemm_params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda,
        ldb,
        ldd: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
        batch_stride_a: batch_stride_a as isize,
        batch_stride_b: batch_stride_b as isize,
        batch_stride_d: (m * n) as isize,
        batch_ndim: 1i32,
        gemm_k_iterations_aligned: (k / bk) as i32,
    };
    let gemm_addmm_params = GEMMAddMMParams {
        ldc: c_stride[c_stride.len() - 2] as i32,
        fdc: c_stride[c_stride.len() - 1] as i32,
        batch_stride_c: (m * n) as isize,
        alpha,
        beta,
    };
    let batch_strides = [
        gemm_params.batch_stride_a,
        gemm_params.batch_stride_b,
        gemm_addmm_params.batch_stride_c,
    ];

    // TODO(laurent): generate the name
    // template [[host_name("gemm_" #tname "_"  #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn)]]
    let name = match (dtype, a_trans, b_trans) {
        (GemmDType::F32, false, false) => "gemm_nn_f32_f32_32_32_16_2_2",
        (GemmDType::F32, true, false) => "gemm_tn_f32_f32_32_32_16_2_2",
        (GemmDType::F32, false, true) => "gemm_nt_f32_f32_32_32_16_2_2",
        (GemmDType::F32, true, true) => "gemm_tt_f32_f32_32_32_16_2_2",
        (GemmDType::BF16, false, false) => "gemm_nn_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, true, false) => "gemm_tn_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, false, true) => "gemm_nt_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, true, true) => "gemm_tt_bf16_bf16_32_32_16_2_2",
        (GemmDType::F16, false, false) => "gemm_nn_f16_f16_32_32_16_2_2",
        (GemmDType::F16, true, false) => "gemm_tn_f16_f16_32_32_16_2_2",
        (GemmDType::F16, false, true) => "gemm_nt_f16_f16_32_32_16_2_2",
        (GemmDType::F16, true, true) => "gemm_tt_f16_f16_32_32_16_2_2",
    };
    let pipeline = kernels.load_pipeline_with_constants(device, Source::Gemm, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as NSUInteger);
    encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as NSUInteger);
    encoder.set_buffer(2, Some(c_buffer), c_offset as NSUInteger);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_bytes(
        4,
        std::mem::size_of::<GemmParams>() as u64,
        &gemm_params as *const GemmParams as *const c_void,
    );
    encoder.set_bytes(
        5,
        std::mem::size_of::<GEMMAddMMParams>() as u64,
        &gemm_addmm_params as *const GEMMAddMMParams as *const c_void,
    );
    encoder.set_bytes(
        6, // batch_shape
        std::mem::size_of::<i32>() as u64,
        &(b as i32) as *const i32 as *const c_void,
    );
    encoder.set_bytes(
        7,
        (std::mem::size_of::<isize>() * batch_strides.len()) as u64,
        batch_strides.as_ptr() as *const c_void,
    );

    let grid_size = MTLSize {
        width: tn as u64,
        height: tm as u64,
        depth: /* batch_size_out */ b as u64,
    };
    let group_size = MTLSize {
        width: 32,
        height: wn,
        depth: wm,
    };
    encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}

pub fn call_const_fill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    length: usize,
    output: &Buffer,
    v: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Fill, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(encoder, (output, v, length));
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_mul_and_act_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    length: usize,
    left: BufferOffset,
    right: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::MulAndAct, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, &left, &right, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.use_resource(left.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(right.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_mul_and_act_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    left_input: BufferOffset,
    left_strides: &[usize],
    right_input: BufferOffset,
    right_strides: &[usize],
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::MulAndAct, name)?;

    let num_dims: usize = shape.len();
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let width: usize = shape.iter().product();
    let length: usize = shape.iter().product();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, width);

    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            left_strides,
            right_strides,
            &left_input,
            &right_input,
            output
        )
    );
    encoder.use_resource(left_input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(right_input.buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

#[cfg(test)]
mod tests;
