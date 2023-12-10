use metal::{
    Buffer, CommandBufferRef, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState,
    Device, Function, FunctionConstantValues, Library, MTLDataType, MTLResourceUsage, MTLSize,
    NSUInteger,
};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::hash::Hash;
use std::io::{stdout, Write};
use std::sync::RwLock;

const AFFINE: &str = include_str!("affine.metal");
const INDEXING: &str = include_str!("indexing.metal");
const UNARY: &str = include_str!("unary.metal");
const BINARY: &str = include_str!("binary.metal");
const TERNARY: &str = include_str!("ternary.metal");
const CAST: &str = include_str!("cast.metal");
const REDUCE: &str = include_str!("reduce.metal");
const MFA_LIB: &[u8] = include_bytes!("mfa.metallib");

fn linear_split(pipeline: &ComputePipelineState, length: usize) -> (MTLSize, MTLSize) {
    let size = length as u64;
    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), size);
    let count = (size + width - 1) / width;
    let thread_group_count = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    (thread_group_count, thread_group_size)
}

fn set_param<P: EncoderParam>(encoder: &ComputeCommandEncoderRef, position: u64, data: P) {
    <P as EncoderParam>::set_param(encoder, position, data)
}
trait EncoderParam {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self);
}
macro_rules! primitive {
    ($type:ty) => {
        impl EncoderParam for $type {
            fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
                encoder.set_bytes(
                    position,
                    core::mem::size_of::<$type>() as u64,
                    &data as *const $type as *const c_void,
                );
            }
        }
    };
}
primitive!(usize);
primitive!(u32);
primitive!(f32);

impl<T> EncoderParam for &[T] {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_bytes(
            position,
            core::mem::size_of_val(data) as u64,
            data.as_ptr() as *const c_void,
        );
    }
}

impl EncoderParam for &Buffer {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}
impl EncoderParam for (&Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1 as u64);
    }
}
impl EncoderParam for &mut Buffer {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}
impl EncoderParam for (&mut Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1 as u64);
    }
}

macro_rules! set_params {
    ($encoder:ident, ($($param:expr),+)) => (
        let mut _index = 0;
        $(
            set_param($encoder, _index, $param);
            _index += 1;
        )*
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Indexing,
    Unary,
    Binary,
    Ternary,
    Cast,
    Reduce,
    MetalFlashAttention,
}

macro_rules! ops{
    ($($name:ident),+) => {

        pub mod contiguous {
        pub struct Kernel(pub &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_float"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_half"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bfloat"));
        }
        )+
            pub mod copy {
                use super::Kernel;
                pub const FLOAT: Kernel = Kernel("copy_float");
                pub const HALF: Kernel = Kernel("copy_half");
                pub const BFLOAT: Kernel = Kernel("copy_bfloat");
                pub const U32: Kernel = Kernel("copy_u32");
                pub const U8: Kernel = Kernel("copy_u8");
            }
        }

        pub mod strided {
        pub struct Kernel(pub &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_float_strided"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_half_strided"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bfloat_strided"));
        }
        )+
            pub mod copy {
                use super::Kernel;
                pub const FLOAT: Kernel = Kernel("copy_float_strided");
                pub const HALF: Kernel = Kernel("copy_half_strided");
                pub const BFLOAT: Kernel = Kernel("copy_bfloat_strided");
                pub const U32: Kernel = Kernel("copy_u32_strided");
                pub const U8: Kernel = Kernel("copy_u8_strided");
            }
        }
    };
}

pub mod unary {
    ops!(cos, sin, exp, sqr, sqrt, neg, log, gelu, ceil, floor, round, erf, gelu_erf);
}
pub mod binary {
    ops!(add, sub, mul, div);
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
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

type KernelMap<T> = HashMap<KernelKey, T>;
type Libraries = HashMap<Source, Library>;
type Pipelines = KernelMap<ComputePipelineState>;

#[derive(Debug, Default)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
}

enum LibraryDefinition {
    Source(&'static str),
    Data(&'static [u8]),
}

impl From<&'static str> for LibraryDefinition {
    fn from(s: &'static str) -> Self {
        Self::Source(s)
    }
}
impl From<&'static [u8]> for LibraryDefinition {
    fn from(s: &'static [u8]) -> Self {
        Self::Data(s)
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

    fn get_library_source(&self, source: Source) -> LibraryDefinition {
        match source {
            Source::Affine => AFFINE.into(),
            Source::Unary => UNARY.into(),
            Source::Binary => BINARY.into(),
            Source::Ternary => TERNARY.into(),
            Source::Indexing => INDEXING.into(),
            Source::Cast => CAST.into(),
            Source::Reduce => REDUCE.into(),
            Source::MetalFlashAttention => MFA_LIB.into(),
        }
    }

    pub fn load_library(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, MetalKernelError> {
        let mut libraries = self.libraries.write()?;
        if let Some(lib) = libraries.get(&source) {
            Ok(lib.clone())
        } else {
            let lib = match self.get_library_source(source) {
                LibraryDefinition::Source(source_content) => device
                    .new_library_with_source(source_content, &CompileOptions::new())
                    .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?,
                LibraryDefinition::Data(data) => device
                    .new_library_with_data(data)
                    .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?,
            };

            libraries.insert(source, lib.clone());
            Ok(lib)
        }
    }

    fn load_function(
        &self,
        device: &Device,
        source: Source,
        key: KernelKey,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device, source)?
            .get_function(
                key.name,
                key.constants.map(|c| c.create_function_constant_values()),
            )
            .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
        Ok(func)
    }

    pub fn load_pipeline<T: Into<KernelKey>>(
        &self,
        device: &Device,
        source: Source,
        key: T,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let key: KernelKey = key.into();
        let mut pipelines = self.pipelines.write()?;
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let func = self.load_function(device, source, key.clone())?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
            pipelines.insert(key, pipeline.clone());

            Ok(pipeline)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct KernelKey {
    name: &'static str,
    constants: Option<ConstantValues>,
}

impl KernelKey {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            constants: None,
        }
    }

    fn with_constants(mut self, constants: ConstantValues) -> Self {
        self.constants = Some(constants);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ConstantValueId {
    Index(NSUInteger),
    Name(&'static str),
}

trait MetalDType {
    const MTL_DATA_TYPE: MTLDataType;
}
macro_rules! metal_dtype {
    ($ty:ty, $mtl_data_type:ident) => {
        impl MetalDType for $ty {
            const MTL_DATA_TYPE: MTLDataType = MTLDataType::$mtl_data_type;
        }
    };
}
metal_dtype!(f32, Float);
metal_dtype!(u32, UInt);
metal_dtype!(u16, UShort);
metal_dtype!(bool, Bool);

#[derive(Debug, Clone, PartialEq)]
enum ConstantValueType {
    Float(f32),
    Uint(u32),
    UShort(u16),
    Bool(bool),
}

impl Hash for ConstantValueType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use ConstantValueType::*;
        match self {
            Float(v) => v.to_bits().hash(state),
            Uint(v) => v.hash(state),
            UShort(v) => v.hash(state),
            Bool(v) => v.hash(state),
        }
    }
}

impl Eq for ConstantValueType {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConstantValues(BTreeMap<ConstantValueId, ConstantValueType>);

macro_rules! add_indexed_constant {
    ($fcv:expr, $value:expr, $ty:ty, $idx:expr) => {
        $fcv.set_constant_value_at_index(
            $value as *const $ty as *const c_void,
            <$ty>::MTL_DATA_TYPE,
            $idx,
        )
    };
}
macro_rules! add_named_constant {
    ($fcv:expr, $value:expr, $ty:ty, $name:expr) => {
        $fcv.set_constant_value_with_name(
            $value as *const $ty as *const c_void,
            <$ty>::MTL_DATA_TYPE,
            $name,
        )
    };
}

impl Hash for ConstantValues {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for (id, value) in &self.0 {
            id.hash(state);
            value.hash(state);
        }
    }
}

impl ConstantValues {
    fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn set(mut self, id: impl Into<ConstantValueId>, value: impl Into<ConstantValueType>) -> Self {
        self.0.insert(id.into(), value.into());
        self
    }

    fn create_function_constant_values(&self) -> FunctionConstantValues {
        use ConstantValueId::*;
        use ConstantValueType::*;
        let mut function_values = FunctionConstantValues::new();

        for (id, value) in &self.0 {
            match (&id, &value) {
                (Index(index), Float(value)) => {
                    add_indexed_constant!(function_values, value, f32, *index);
                }
                (Index(index), Uint(value)) => {
                    add_indexed_constant!(function_values, value, u32, *index);
                }
                (Index(index), UShort(value)) => {
                    add_indexed_constant!(function_values, value, u16, *index);
                }
                (Index(index), Bool(value)) => {
                    add_indexed_constant!(function_values, value, bool, *index);
                }
                (Name(name), Float(value)) => {
                    add_named_constant!(function_values, value, f32, name);
                }
                (Name(name), Uint(value)) => {
                    add_named_constant!(function_values, value, u32, name);
                }
                (Name(name), UShort(value)) => {
                    add_named_constant!(function_values, value, u16, name);
                }
                (Name(name), Bool(value)) => {
                    add_named_constant!(function_values, value, bool, name);
                }
            }
        }
        function_values
    }
}

impl From<&'static str> for KernelKey {
    fn from(name: &'static str) -> Self {
        Self {
            name,
            constants: None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_unary_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: unary::contiguous::Kernel,
    length: usize,
    input: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel_name.0)?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_unary_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: unary::strided::Kernel,
    shape: &[usize],
    input: &Buffer,
    strides: &[usize],
    offset: usize,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, name.0)?;

    let num_dims: usize = shape.len();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            strides,
            (input, offset),
            (output, output_offset)
        )
    );

    let width: usize = shape.iter().product();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, width);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_binary_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: binary::contiguous::Kernel,
    length: usize,
    left: &Buffer,
    right: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Binary, kernel_name.0)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, left, right, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_binary_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: binary::strided::Kernel,
    shape: &[usize],
    left_input: &Buffer,
    left_strides: &[usize],
    left_offset: usize,
    right_input: &Buffer,
    right_strides: &[usize],
    right_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Binary, name.0)?;

    let num_dims: usize = shape.len();
    let encoder = command_buffer.new_compute_command_encoder();
    let width: usize = shape.iter().product();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();

    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            left_strides,
            right_strides,
            (left_input, left_offset),
            (right_input, right_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, width);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_cast_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Cast, kernel_name)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, (input, input_offset), output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_cast_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    input: &Buffer,
    input_strides: &[usize],
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Cast, kernel_name)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();

    set_params!(
        encoder,
        (
            length,
            shape.len(),
            shape,
            input_strides,
            (input, input_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

pub fn call_reduce_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    out_length: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let elements_to_sum = length / out_length;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (length, elements_to_sum, (input, input_offset), output)
    );

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

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_last_softmax(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    input: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, elements_to_sum, input, output));

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

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_affine(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    size: usize,
    input: &Buffer,
    output: &Buffer,
    mul: f32,
    add: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (size, mul, add, input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_affine_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: &Buffer,
    input_stride: &[usize],
    input_offset: usize,
    output: &Buffer,
    mul: f32,
    add: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
    let size: usize = shape.iter().product();

    let encoder = command_buffer.new_compute_command_encoder();
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
            (input, input_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

pub fn call_where_cond_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    cond: &Buffer,
    (cond_stride, cond_offset): (&[usize], usize),
    left: &Buffer,
    (left_stride, left_offset): (&[usize], usize),
    right: &Buffer,
    (right_stride, right_offset): (&[usize], usize),
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Ternary, name)?;

    let encoder = command_buffer.new_compute_command_encoder();
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
            (cond, cond_offset),
            (left, left_offset),
            (right, right_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_index_select(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    ids_size: usize,
    dim: usize,
    input: &Buffer,
    ids: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let src_dim_size = shape[dim];
    let dst_el = ids_size * left_size * right_size;

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            ids_size,
            input,
            ids,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

impl From<NSUInteger> for ConstantValueId {
    fn from(idx: NSUInteger) -> Self {
        Self::Index(idx)
    }
}

impl From<usize> for ConstantValueId {
    fn from(idx: usize) -> Self {
        ConstantValueId::from(idx as NSUInteger)
    }
}

impl From<i32> for ConstantValueId {
    fn from(idx: i32) -> Self {
        ConstantValueId::from(idx as NSUInteger)
    }
}

impl From<&'static str> for ConstantValueId {
    fn from(name: &'static str) -> Self {
        Self::Name(name)
    }
}

macro_rules! to_constant_value {
    ($ty:ty, $constant_value_type:ident) => {
        to_constant_value!($ty, $ty, $constant_value_type);
    };
    ($ty:ty, $via:ty, $constant_value_type:ident) => {
        impl From<$ty> for ConstantValueType {
            fn from(v: $ty) -> Self {
                Self::$constant_value_type(v as $via)
            }
        }
    };
}
to_constant_value!(f32, Float);
to_constant_value!(u32, Uint);
to_constant_value!(usize, u32, Uint);
to_constant_value!(u16, UShort);
to_constant_value!(bool, Bool);

struct MFAGemmConfig {
    m: usize,
    k: usize,
    n: usize,
    transpose_left: bool,
    transpose_right: bool,
    batched: bool,
    m_simd: u16,
    n_simd: u16,
    k_simd: u16,
    m_splits: u16,
    n_splits: u16,
    m_group: u16,
    n_group: u16,
}

impl From<MFAGemmConfig> for ConstantValues {
    fn from(conf: MFAGemmConfig) -> Self {
        ConstantValues::new()
            .set(0, conf.m)
            .set(1, conf.k)
            .set(2, conf.n)
            .set(10, conf.transpose_left)
            .set(11, conf.transpose_right)
            .set(12, false)
            .set(20, 1.0)
            .set(21, 0.0)
            .set(100, conf.batched)
            .set(101, false)
            .set(50001, false)
            .set(200, conf.m_simd)
            .set(201, conf.n_simd)
            .set(202, conf.k_simd)
            .set(210, conf.m_splits)
            .set(211, conf.n_splits)
            // garbage
            .set(102, false)
            .set(103, false)
            .set(113, false)
            .set(50000, false)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_mfa_gemm(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    lhs: &Buffer,
    lhs_dims: &[usize],
    rhs: &Buffer,
    rhs_dims: &[usize],
    output: &Buffer,
    (b, m, n, k): (usize, usize, usize, usize),
    transpose_left: bool,
    transpose_right: bool,
) -> Result<(), MetalKernelError> {
    let batched = b > 1;

    let mut c_elements = m * n;
    if batched {
        c_elements *= 2;
    }

    let is_half = name == "hgemm";
    let is_float = name == "sgemm";

    let mut m_group = 32;
    let mut n_group = 32;
    let mut k_simd = 32;
    if c_elements > 10 ^ 6 {
        m_group = 48;
        n_group = 48;
    }
    // If K_simd is perfectly equal to matrix K, the compiler can elide a large
    // amount of logic in the kernel.
    if k >= 33 && k <= 40 {
        k_simd = 40;
    } else if is_half && k >= 73 && k >= 80 {
        k_simd = 80;
    } else if c_elements > 10 ^ 6 {
        if k <= 16 {
            k_simd = 16;
        } else if k <= 24 {
            k_simd = 24;
        } else if k <= 32 {
            k_simd = 32;
        } else if k <= 48 {
            k_simd = 24;
        } else if k <= 64 {
            k_simd = 32;
        } else if is_float {
            k_simd = 24;
        }
    }

    let m_splits = 2;
    let n_splits = 2;
    let m_simd = m_group / m_splits;
    let n_simd = n_group / n_splits;

    let config = MFAGemmConfig {
        m,
        k,
        n,
        transpose_left,
        transpose_right,
        batched,
        m_simd,
        n_simd,
        k_simd,
        m_splits,
        n_splits,
        m_group,
        n_group,
    };

    let pipeline = kernels.load_pipeline(
        device,
        Source::MetalFlashAttention,
        KernelKey::new(name).with_constants(config.into()),
    )?;
    let block_type_size = if is_half { 2 } else { 4 };
    let a_block_bytes = m_group * k_simd * block_type_size;
    let b_block_bytes = k_simd * n_group * block_type_size;
    let c_block_bytes = m_group * n_group * block_type_size;
    let mut thread_group_memory_length = a_block_bytes + b_block_bytes;

    if m % 8 > 0 && n % 8 > 0 {
        thread_group_memory_length = core::cmp::max(thread_group_memory_length, c_block_bytes);
    }

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_threadgroup_memory_length(0, thread_group_memory_length as NSUInteger);
    encoder.use_resources(&[&lhs, &rhs], MTLResourceUsage::Read);
    encoder.use_resource(&output, MTLResourceUsage::Write);
    encoder.set_buffers(0, &[Some(lhs), Some(rhs), Some(output)], &[0; 3]);

    let ceil_divide = |a, b| (a + b - 1) / b;

    let mut grid_z = 1;

    if batched {
        grid_z = lhs_dims[..lhs_dims.len() - 2].iter().product();
        let byte_stride = |shape: &[usize]| -> u64 {
            let rank = shape.len();
            let mut output = core::mem::size_of::<f32>() * shape[rank - 2] * shape[rank - 1];
            if shape[..shape.len() - 2].iter().product::<usize>() == 1 {
                output = 0;
            }
            output as u64
        };
        let byte_stride_a = byte_stride(lhs_dims);
        let byte_stride_b = byte_stride(rhs_dims);
        let byte_stride_c = byte_stride(&[m, n]);

        type BatchConfig = (u64, u64, u64, u64);
        let mut batching_conf: Vec<BatchConfig> = vec![];
        for i in 0..grid_z {
            batching_conf.push((
                i as u64 * byte_stride_a,
                i as u64 * byte_stride_b,
                i as u64 * byte_stride_c,
                0u64,
            ));
        }
        set_param(encoder, 10, batching_conf.as_slice());
    }

    let grid_size = MTLSize::new(
        ceil_divide(n as NSUInteger, n_group as NSUInteger),
        ceil_divide(m as NSUInteger, m_group as NSUInteger),
        grid_z as NSUInteger,
    );

    let group_size = MTLSize::new((32 * m_splits * n_splits) as NSUInteger, 1, 1);
    encoder.dispatch_thread_groups(grid_size, group_size);
    encoder.end_encoding();
    Ok(())
}

#[cfg(test)]
mod tests;
