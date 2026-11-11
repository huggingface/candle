use crate::App;
use glam::IVec2;
use rand::Rng;
use std::{slice, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorBufferInfo, DescriptorImageInfo,
        DescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, ComputePipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
};

/// Pipeline holding double buffered grid & color image. Grids are used to calculate the state, and
/// color image is used to show the output. Because on each step we determine state in parallel, we
/// need to write the output to another grid. Otherwise the state would not be correctly determined
/// as one shader invocation might read data that was just written by another shader invocation.
pub struct GameOfLifeComputePipeline {
    compute_queue: Arc<Queue>,
    compute_life_pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    life_in: Subbuffer<[u32]>,
    life_out: Subbuffer<[u32]>,
    image: Arc<ImageView>,
}

fn rand_grid(memory_allocator: &Arc<StandardMemoryAllocator>, size: [u32; 2]) -> Subbuffer<[u32]> {
    Buffer::from_iter(
        memory_allocator,
        &BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (0..(size[0] * size[1])).map(|_| rand::thread_rng().gen_range(0u32..=1)),
    )
    .unwrap()
}

impl GameOfLifeComputePipeline {
    pub fn new(app: &App, compute_queue: &Arc<Queue>, size: [u32; 2]) -> GameOfLifeComputePipeline {
        let memory_allocator = app.context.memory_allocator();
        let life_in = rand_grid(memory_allocator, size);
        let life_out = rand_grid(memory_allocator, size);

        let compute_life_pipeline = {
            let device = compute_queue.device();
            let cs = unsafe { compute_life_cs::load(device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(&cs);
            let layout = PipelineLayout::from_stages(device, slice::from_ref(&stage)).unwrap();

            ComputePipeline::new(
                device,
                None,
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        let image = ImageView::new_default(
            &Image::new(
                memory_allocator,
                &ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R8G8B8A8_UNORM,
                    extent: [size[0], size[1], 1],
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED | ImageUsage::STORAGE,
                    ..Default::default()
                },
                &AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        GameOfLifeComputePipeline {
            compute_queue: compute_queue.clone(),
            compute_life_pipeline,
            command_buffer_allocator: app.command_buffer_allocator.clone(),
            descriptor_set_allocator: app.descriptor_set_allocator.clone(),
            life_in,
            life_out,
            image,
        }
    }

    pub fn color_image(&self) -> Arc<ImageView> {
        self.image.clone()
    }

    pub fn draw_life(&self, pos: IVec2) {
        let mut life_in = self.life_in.write().unwrap();
        let extent = self.image.image().extent();
        if pos.y < 0 || pos.y >= extent[1] as i32 || pos.x < 0 || pos.x >= extent[0] as i32 {
            return;
        }
        let index = (pos.y * extent[0] as i32 + pos.x) as usize;
        life_in[index] = 1;
    }

    pub fn compute(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        life_color: [f32; 4],
        dead_color: [f32; 4],
    ) -> Box<dyn GpuFuture> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.compute_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Dispatch will mutate the builder adding commands which won't be sent before we build the
        // command buffer after dispatches. This will minimize the commands we send to the GPU. For
        // example, we could be doing tens of dispatches here depending on our needs. Maybe we
        // wanted to simulate 10 steps at a time...

        // First compute the next state.
        self.dispatch(&mut builder, life_color, dead_color, 0);

        // Then color based on the next state.
        self.dispatch(&mut builder, life_color, dead_color, 1);

        let command_buffer = builder.build().unwrap();
        let finished = before_future
            .then_execute(self.compute_queue.clone(), command_buffer)
            .unwrap();
        let after_pipeline = finished.then_signal_fence_and_flush().unwrap().boxed();

        // Swap input and output so the output becomes the input for next frame.
        std::mem::swap(&mut self.life_in, &mut self.life_out);

        after_pipeline
    }

    /// Builds the command for a dispatch.
    fn dispatch(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        life_color: [f32; 4],
        dead_color: [f32; 4],
        // Step determines whether we color or compute life (see branch in the shader)s.
        step: i32,
    ) {
        // Resize image if needed.
        let image_extent = self.image.image().extent();
        let layout = &self.compute_life_pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            &self.descriptor_set_allocator,
            layout,
            &[
                WriteDescriptorSet::image(
                    0,
                    &DescriptorImageInfo {
                        image_view: Some(&self.image),
                        ..Default::default()
                    },
                ),
                WriteDescriptorSet::buffer(
                    1,
                    &DescriptorBufferInfo {
                        buffer: Some(self.life_in.buffer()),
                        ..Default::default()
                    },
                ),
                WriteDescriptorSet::buffer(
                    2,
                    &DescriptorBufferInfo {
                        buffer: Some(self.life_out.buffer()),
                        ..Default::default()
                    },
                ),
            ],
            &[],
        )
        .unwrap();

        let push_constants = compute_life_cs::PushConstants {
            life_color,
            dead_color,
            step,
        };
        builder
            .bind_pipeline_compute(self.compute_life_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_life_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(
                self.compute_life_pipeline.layout().clone(),
                0,
                push_constants,
            )
            .unwrap();
        unsafe { builder.dispatch([image_extent[0] / 8, image_extent[1] / 8, 1]) }.unwrap();
    }
}

mod compute_life_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
            layout(set = 0, binding = 1) buffer LifeInBuffer { uint life_in[]; };
            layout(set = 0, binding = 2) buffer LifeOutBuffer { uint life_out[]; };

            layout(push_constant) uniform PushConstants {
                vec4 life_color;
                vec4 dead_color;
                int step;
            } push_constants;

            int get_index(ivec2 pos) {
                ivec2 dims = ivec2(imageSize(img));
                return pos.y * dims.x + pos.x;
            }

            // https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
            void compute_life() {
                ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
                int index = get_index(pos);

                ivec2 up_left = pos + ivec2(-1, 1);
                ivec2 up = pos + ivec2(0, 1);
                ivec2 up_right = pos + ivec2(1, 1);
                ivec2 right = pos + ivec2(1, 0);
                ivec2 down_right = pos + ivec2(1, -1);
                ivec2 down = pos + ivec2(0, -1);
                ivec2 down_left = pos + ivec2(-1, -1);
                ivec2 left = pos + ivec2(-1, 0);

                int alive_count = 0;
                if (life_in[get_index(up_left)] == 1) { alive_count += 1; }
                if (life_in[get_index(up)] == 1) { alive_count += 1; }
                if (life_in[get_index(up_right)] == 1) { alive_count += 1; }
                if (life_in[get_index(right)] == 1) { alive_count += 1; }
                if (life_in[get_index(down_right)] == 1) { alive_count += 1; }
                if (life_in[get_index(down)] == 1) { alive_count += 1; }
                if (life_in[get_index(down_left)] == 1) { alive_count += 1; }
                if (life_in[get_index(left)] == 1) { alive_count += 1; }

                // Dead becomes alive.
                if (life_in[index] == 0 && alive_count == 3) {
                    life_out[index] = 1;
                }
                // Becomes dead.
                else if (life_in[index] == 1 && alive_count < 2 || alive_count > 3) {
                    life_out[index] = 0;
                }
                // Else do nothing.
                else {
                    life_out[index] = life_in[index];
                }
            }

            void compute_color() {
                ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
                int index = get_index(pos);
                if (life_out[index] == 1) {
                    imageStore(img, pos, push_constants.life_color);
                } else {
                    imageStore(img, pos, push_constants.dead_color);
                }
            }

            void main() {
                if (push_constants.step == 0) {
                    compute_life();
                } else {
                    compute_color();
                }
            }
        ",
    }
}
