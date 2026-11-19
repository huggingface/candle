use crate::{App, RenderContext, ShaderLanguage};
use core::slice;
use std::sync::Arc;
use vulkano::{
    image::{mip_level_extent, Image},
    pipeline::{
        compute::ComputePipelineCreateInfo, ComputePipeline, PipelineShaderStageCreateInfo,
    },
    sync::{AccessFlags, PipelineStages},
};
use vulkano_taskgraph::{
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
    Id, Task, TaskContext, TaskResult,
};

const THRESHOLD: f32 = 1.5;
const KNEE: f32 = 0.1;
const INTENSITY: f32 = 1.0;

pub struct BloomTask {
    downsample_pipeline: Arc<ComputePipeline>,
    upsample_pipeline: Arc<ComputePipeline>,
    bloom_image_id: Id<Image>,
}

impl BloomTask {
    pub fn new(app: &App, virtual_bloom_image_id: Id<Image>) -> Self {
        let bcx = app.resources.bindless_context().unwrap();

        let downsample_pipeline = {
            let cs = match app.shader_language {
                ShaderLanguage::Glsl => unsafe { downsample_glsl::load(&app.device) }
                    .unwrap()
                    .entry_point("main"),
                ShaderLanguage::Slang => unsafe { downsample_slang::load(&app.device) }
                    .unwrap()
                    .entry_point("main"),
            }
            .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(&cs);
            let layout = bcx
                .pipeline_layout_from_stages(slice::from_ref(&stage))
                .unwrap();

            ComputePipeline::new(
                &app.device,
                None,
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        let upsample_pipeline = {
            let cs = match app.shader_language {
                ShaderLanguage::Glsl => unsafe { upsample_glsl::load(&app.device) }
                    .unwrap()
                    .entry_point("main"),
                ShaderLanguage::Slang => unsafe { upsample_slang::load(&app.device) }
                    .unwrap()
                    .entry_point("main"),
            }
            .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(&cs);
            let layout = bcx
                .pipeline_layout_from_stages(slice::from_ref(&stage))
                .unwrap();

            ComputePipeline::new(
                &app.device,
                None,
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        BloomTask {
            downsample_pipeline,
            upsample_pipeline,
            bloom_image_id: virtual_bloom_image_id,
        }
    }
}

impl Task for BloomTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let bloom_image = tcx.image(self.bloom_image_id).image();

        let dependency_info = DependencyInfo {
            memory_barriers: &[MemoryBarrier {
                src_stages: PipelineStages::COMPUTE_SHADER,
                src_access: AccessFlags::SHADER_WRITE,
                dst_stages: PipelineStages::COMPUTE_SHADER,
                dst_access: AccessFlags::SHADER_READ,
                ..Default::default()
            }],
            ..Default::default()
        };

        cbf.bind_pipeline(&self.downsample_pipeline);

        for src_mip_level in 0..bloom_image.mip_levels() - 1 {
            let dst_mip_level = src_mip_level + 1;
            let dst_extent = mip_level_extent(bloom_image.extent(), dst_mip_level).unwrap();
            let group_counts = dst_extent.map(|c| c.div_ceil(8));

            cbf.push_constants(
                self.downsample_pipeline.layout(),
                0,
                // We use the GLSL shader's push constants regardless of which shader is used for
                // simplicity because we know that they have the same representation.
                &downsample_glsl::PushConstants {
                    sampler_id: rcx.bloom_sampler_id,
                    texture_id: rcx.bloom_sampled_image_id,
                    dst_mip_image_id: rcx.bloom_storage_image_ids[dst_mip_level as usize],
                    dst_mip_level,
                    threshold: THRESHOLD,
                    knee: KNEE,
                },
            );

            unsafe { cbf.dispatch(group_counts) };

            cbf.pipeline_barrier(&dependency_info);
        }

        cbf.bind_pipeline(&self.upsample_pipeline);

        for dst_mip_level in (0..bloom_image.mip_levels() - 1).rev() {
            let dst_extent = mip_level_extent(bloom_image.extent(), dst_mip_level).unwrap();
            let group_counts = dst_extent.map(|c| c.div_ceil(8));

            cbf.push_constants(
                self.upsample_pipeline.layout(),
                0,
                // We use the GLSL shader's push constants regardless of which shader is used for
                // simplicity because we know that they have the same representation.
                &upsample_glsl::PushConstants {
                    sampler_id: rcx.bloom_sampler_id,
                    texture_id: rcx.bloom_sampled_image_id,
                    dst_mip_image_id: rcx.bloom_storage_image_ids[dst_mip_level as usize],
                    dst_mip_level,
                    intensity: INTENSITY,
                },
            );

            unsafe { cbf.dispatch(group_counts) };

            if dst_mip_level != 0 {
                cbf.pipeline_barrier(&dependency_info);
            }
        }

        Ok(())
    }
}

mod downsample_glsl {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/downsample.glsl",
    }
}

mod upsample_glsl {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/upsample.glsl",
    }
}

mod downsample_slang {
    vulkano_shaders::shader! {
        ty: "compute",
        lang: "slang",
        entry_point: "downsample",
        path: "shaders/bloom.slang",
    }
}

mod upsample_slang {
    vulkano_shaders::shader! {
        ty: "compute",
        lang: "slang",
        entry_point: "upsample",
        path: "shaders/bloom.slang",
    }
}
