use crate::utils::EncoderProvider;
use crate::{
    debug_group, set_params, BufferOffset, ComputeCommandEncoder, Device, GgmlDType, Kernels,
    MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;
use std::mem::size_of;

const MAX_MV_TOKENS: usize = 8;
const ROUTE_THREADS: usize = 128;
const ROUTE_COUNTS: usize = ROUTE_THREADS + 1;
const ROUTE_TILE: usize = 32;
const OUTPUT_TILE: usize = 64;
const MATRIX_THREADGROUP_BYTES: usize = 8192 + ROUTE_TILE * 2 * size_of::<u32>();
const TASK_BYTES: usize = 4 * size_of::<u32>();

#[derive(Clone, Copy, Debug)]
pub struct QuantizedMoeShape {
    pub dtype: GgmlDType,
    pub experts: usize,
    pub tokens: usize,
    pub selected_experts: usize,
    pub input_slots: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub weight_row_bytes: usize,
    pub weight_expert_bytes: usize,
}

pub struct QuantizedMoeBuffers<'a> {
    pub weights: BufferOffset<'a>,
    pub input: BufferOffset<'a>,
    pub ids: BufferOffset<'a>,
    pub output: BufferOffset<'a>,
    pub scratch: Option<QuantizedMoeScratch<'a>>,
}

pub struct QuantizedMoeScratch<'a> {
    pub counts: BufferOffset<'a>,
    pub routes: BufferOffset<'a>,
    pub tasks: BufferOffset<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizedMoeScratchSizes {
    pub counts: usize,
    pub routes: usize,
    pub tasks: usize,
}

struct FormatKernels {
    vector: &'static str,
    matrix: &'static str,
    threads: (usize, usize),
    output_divisor: usize,
}

fn format_kernels(dtype: GgmlDType) -> Result<FormatKernels, MetalKernelError> {
    macro_rules! entry {
        ($suffix:literal, $x:literal, $y:literal, $divisor:literal) => {
            FormatKernels {
                vector: concat!("kernel_mul_mv_id_", $suffix, "_f32"),
                matrix: concat!("kernel_mul_mm_routed_", $suffix, "_f32"),
                threads: ($x, $y),
                output_divisor: $divisor,
            }
        };
    }
    Ok(match dtype {
        GgmlDType::Q4_0 => entry!("q4_0", 8, 8, 8),
        GgmlDType::Q4_1 => entry!("q4_1", 8, 8, 8),
        GgmlDType::Q5_0 => entry!("q5_0", 8, 8, 8),
        GgmlDType::Q5_1 => entry!("q5_1", 8, 8, 8),
        GgmlDType::Q8_0 => entry!("q8_0", 8, 8, 8),
        GgmlDType::Q2K => entry!("q2_K", 2, 32, 8),
        GgmlDType::Q3K => entry!("q3_K", 2, 32, 4),
        GgmlDType::Q4K => entry!("q4_K", 4, 8, 4),
        GgmlDType::Q5K => entry!("q5_K", 2, 32, 4),
        GgmlDType::Q6K => entry!("q6_K", 2, 32, 2),
        GgmlDType::F16 => entry!("f16", 32, 1, 1),
        GgmlDType::F32 => entry!("f32", 32, 1, 1),
        GgmlDType::BF16 | GgmlDType::Q8_1 | GgmlDType::Q8K => {
            return Err(MetalKernelError::InvalidInput(format!(
                "indexed MoE does not support {dtype:?} weights"
            )))
        }
    })
}

fn overflow() -> MetalKernelError {
    MetalKernelError::InvalidInput("indexed MoE workspace size overflow".to_owned())
}

impl QuantizedMoeShape {
    fn use_vector(&self, kernels: &FormatKernels) -> bool {
        self.tokens <= MAX_MV_TOKENS
            && self.output_dim.is_multiple_of(kernels.output_divisor)
            && self.input_dim >= kernels.threads.0 * kernels.threads.1
            && self.input_dim <= i32::MAX as usize
            && self.experts <= i32::MAX as usize
            && self.selected_experts <= i32::MAX as usize
            && self.weight_expert_bytes <= u32::MAX as usize
    }

    fn pairs(&self) -> Result<usize, MetalKernelError> {
        self.tokens
            .checked_mul(self.selected_experts)
            .ok_or_else(overflow)
    }

    fn task_capacity(&self) -> Result<usize, MetalKernelError> {
        let pairs = self.pairs()?;
        Ok(pairs.min(
            pairs
                .div_ceil(ROUTE_TILE)
                .checked_add(self.experts)
                .ok_or_else(overflow)?,
        ))
    }

    /// Scratch sizes are bytes; the vector path needs no workspace.
    pub fn scratch_sizes(&self) -> Result<Option<QuantizedMoeScratchSizes>, MetalKernelError> {
        let kernels = format_kernels(self.dtype)?;
        if self.tokens == 0 || self.selected_experts == 0 || self.use_vector(&kernels) {
            return Ok(None);
        }
        let counts = self
            .experts
            .checked_add(1)
            .and_then(|n| n.checked_mul(ROUTE_COUNTS))
            .and_then(|n| n.checked_mul(size_of::<u32>()))
            .ok_or_else(overflow)?;
        let routes = self
            .pairs()?
            .checked_mul(size_of::<u32>())
            .ok_or_else(overflow)?;
        let tasks = self
            .task_capacity()?
            .checked_add(1)
            .and_then(|n| n.checked_mul(TASK_BYTES))
            .ok_or_else(overflow)?;
        Ok(Some(QuantizedMoeScratchSizes {
            counts,
            routes,
            tasks,
        }))
    }
}

pub fn call_quantized_moe(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    shape: &QuantizedMoeShape,
    buffers: QuantizedMoeBuffers<'_>,
) -> Result<(), MetalKernelError> {
    let format = format_kernels(shape.dtype)?;
    if shape.tokens == 0 || shape.selected_experts == 0 {
        return Ok(());
    }
    if shape.use_vector(&format) {
        let pipeline = kernels.load_pipeline(device, Source::Quantized, format.vector)?;
        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        debug_group!(encoder, "quantized_moe_mv {:?}", shape);
        set_params!(
            encoder,
            (
                &buffers.weights,
                &buffers.input,
                Output::with_offset(buffers.output.buffer, buffers.output.offset_in_bytes),
                &buffers.ids,
                shape.selected_experts as i64,
                shape.tokens as i64,
                (shape.selected_experts * size_of::<u32>()) as u64,
                shape.input_dim as i64,
                shape.output_dim as i64,
                shape.experts as i64,
                0u64,
                shape.weight_row_bytes as u64,
                shape.weight_expert_bytes as u64,
                shape.input_dim as i64,
                shape.input_slots as i64,
                shape.tokens as i64,
                1i64,
                size_of::<f32>() as u64,
                (shape.input_dim * size_of::<f32>()) as u64,
                (shape.input_slots * shape.input_dim * size_of::<f32>()) as u64,
                shape.output_dim as i64,
                shape.selected_experts as i64,
                (shape.output_dim * size_of::<f32>()) as u64
            )
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: shape.output_dim / format.output_divisor,
                height: 1,
                depth: shape.pairs()?,
            },
            MTLSize {
                width: format.threads.0,
                height: format.threads.1,
                depth: 1,
            },
        );
        return Ok(());
    }

    let scratch = buffers.scratch.as_ref().ok_or_else(|| {
        MetalKernelError::InvalidInput("indexed MoE matrix path requires workspace".to_owned())
    })?;
    let count_pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_moe_route_counts")?;
    let compact_pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_moe_route_compact")?;
    let matrix_pipeline = kernels.load_pipeline(device, Source::Quantized, format.matrix)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    debug_group!(encoder, "quantized_moe_routed {:?}", shape);
    let groups = MTLSize {
        width: shape.experts + 1,
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width: ROUTE_THREADS,
        height: 1,
        depth: 1,
    };
    let pairs = shape.pairs()? as u32;
    encoder.set_compute_pipeline_state(&count_pipeline);
    set_params!(
        encoder,
        (
            &buffers.ids,
            Output::with_offset(scratch.counts.buffer, scratch.counts.offset_in_bytes),
            pairs,
            shape.experts as u32
        )
    );
    encoder.dispatch_thread_groups(groups, threads);

    encoder.set_compute_pipeline_state(&compact_pipeline);
    set_params!(
        encoder,
        (
            &buffers.ids,
            &scratch.counts,
            Output::with_offset(scratch.routes.buffer, scratch.routes.offset_in_bytes),
            Output::with_offset(scratch.tasks.buffer, scratch.tasks.offset_in_bytes),
            pairs,
            shape.experts as u32
        )
    );
    encoder.dispatch_thread_groups(groups, threads);

    encoder.set_compute_pipeline_state(&matrix_pipeline);
    set_params!(
        encoder,
        (
            &buffers.weights,
            &buffers.input,
            &scratch.routes,
            &scratch.tasks,
            Output::with_offset(buffers.output.buffer, buffers.output.offset_in_bytes),
            shape.input_dim as u32,
            shape.output_dim as u32,
            shape.selected_experts as u32,
            shape.input_slots as u32,
            shape.experts as u32,
            shape.weight_row_bytes as u64,
            shape.weight_expert_bytes as u64
        )
    );
    encoder.set_threadgroup_memory_length(0, MATRIX_THREADGROUP_BYTES);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: shape.task_capacity()?,
            height: shape.output_dim.div_ceil(OUTPUT_TILE),
            depth: 1,
        },
        threads,
    );
    Ok(())
}
