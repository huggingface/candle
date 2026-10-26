use super::QMetalStorage;
use crate::backend::BackendStorage;
use crate::quantized::GgmlDType;
use crate::{DType, Layout, MetalStorage, Result, Shape};

const INDEXED_MOE_K_TILE: usize = 32;

impl QMetalStorage {
    pub(crate) fn indexed_moe_forward(
        &self,
        self_shape: &Shape,
        input: &MetalStorage,
        input_l: &Layout,
        ids: &MetalStorage,
        ids_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;
        use candle_metal_kernels::{
            BufferOffset, QuantizedMoeBuffers, QuantizedMoeScratch, QuantizedMoeShape,
        };

        if input.dtype() != DType::F32 || ids.dtype() != DType::U32 {
            crate::bail!("Metal indexed MoE requires F32 input and U32 expert ids")
        }
        if !input_l.is_contiguous() || !ids_l.is_contiguous() {
            crate::bail!("Metal indexed MoE requires contiguous input and expert ids")
        }
        if self.device.id() != input.device().id() || self.device.id() != ids.device().id() {
            crate::bail!("Metal indexed MoE tensors must share one device")
        }
        if matches!(
            self.dtype,
            GgmlDType::Q8_1 | GgmlDType::Q8K | GgmlDType::BF16
        ) {
            crate::bail!(
                "Metal indexed MoE does not support {:?} weights",
                self.dtype
            )
        }

        let (experts, output_dim, input_dim) = self_shape.dims3()?;
        let (tokens, input_slots, input_k) = input_l.shape().dims3()?;
        let (id_tokens, selected_experts) = ids_l.shape().dims2()?;
        if input_k != input_dim || tokens != id_tokens {
            crate::bail!("Metal indexed MoE input, weight, and routing dimensions do not match")
        }
        if input_slots != 1 && input_slots != selected_experts {
            crate::bail!("Metal indexed MoE input slots must be one or match the routing width")
        }
        if experts == 0 || output_dim == 0 || input_dim == 0 {
            crate::bail!("Metal indexed MoE requires nonempty expert weights")
        }
        if !input_dim.is_multiple_of(INDEXED_MOE_K_TILE) {
            crate::bail!("Metal indexed MoE input width must be divisible by 32")
        }
        if experts >= u32::MAX as usize
            || [tokens, selected_experts, input_slots, input_dim, output_dim]
                .into_iter()
                .any(|dim| dim > u32::MAX as usize)
        {
            crate::bail!("Metal indexed MoE dimensions exceed the kernel index range")
        }

        let overflow = || crate::Error::msg("Metal indexed MoE dimensions overflow");
        let routes = tokens.checked_mul(selected_experts).ok_or_else(overflow)?;
        if routes > u32::MAX as usize {
            crate::bail!("Metal indexed MoE route count exceeds the kernel index range")
        }
        let output_elements = routes.checked_mul(output_dim).ok_or_else(overflow)?;
        output_elements
            .checked_mul(DType::F32.size_in_bytes())
            .ok_or_else(overflow)?;
        let weight_row_bytes = (input_dim / self.dtype.block_size())
            .checked_mul(self.dtype.type_size())
            .ok_or_else(overflow)?;
        let weight_expert_bytes = output_dim
            .checked_mul(weight_row_bytes)
            .ok_or_else(overflow)?;
        let weight_bytes = experts
            .checked_mul(weight_expert_bytes)
            .ok_or_else(overflow)?;
        if weight_bytes > self.buffer.length() {
            crate::bail!("Metal indexed MoE weight shape exceeds its storage")
        }
        let input_offset = input_l.start_offset() * DType::F32.size_in_bytes();
        let ids_offset = ids_l.start_offset() * DType::U32.size_in_bytes();
        let output_shape = Shape::from((tokens, selected_experts, output_dim));
        let output = self
            .device
            .new_buffer_builder()
            .with_size_for(output_elements, DType::F32)
            .with_label("indexed_moe")
            .build()?;
        if routes != 0 {
            let shape = QuantizedMoeShape {
                dtype: self.dtype.into(),
                experts,
                tokens,
                selected_experts,
                input_slots,
                input_dim,
                output_dim,
                weight_row_bytes,
                weight_expert_bytes,
            };
            let scratch = match shape.scratch_sizes().map_err(MetalError::from)? {
                Some(sizes) => Some((
                    self.device
                        .new_buffer_builder()
                        .with_size(sizes.counts)
                        .with_label("indexed_moe_counts")
                        .build()?,
                    self.device
                        .new_buffer_builder()
                        .with_size(sizes.routes)
                        .with_label("indexed_moe_routes")
                        .build()?,
                    self.device
                        .new_buffer_builder()
                        .with_size(sizes.tasks)
                        .with_label("indexed_moe_tasks")
                        .build()?,
                )),
                None => None,
            };
            let buffers = QuantizedMoeBuffers {
                weights: BufferOffset::zero_offset(&self.buffer),
                input: BufferOffset {
                    buffer: input.buffer(),
                    offset_in_bytes: input_offset,
                },
                ids: BufferOffset {
                    buffer: ids.buffer(),
                    offset_in_bytes: ids_offset,
                },
                output: BufferOffset::zero_offset(&output),
                scratch: scratch
                    .as_ref()
                    .map(|(counts, routes, tasks)| QuantizedMoeScratch {
                        counts: BufferOffset::zero_offset(counts),
                        routes: BufferOffset::zero_offset(routes),
                        tasks: BufferOffset::zero_offset(tasks),
                    }),
            };
            let encoder = self.device.command_encoder()?;
            candle_metal_kernels::call_quantized_moe(
                self.device.device(),
                &encoder,
                self.device.kernels(),
                &shape,
                buffers,
            )
            .map_err(MetalError::from)?;
        }
        Ok((
            MetalStorage::new(output, self.device.clone(), output_elements, DType::F32),
            output_shape,
        ))
    }
}
