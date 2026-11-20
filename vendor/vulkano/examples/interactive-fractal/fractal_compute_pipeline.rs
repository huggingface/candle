use glam::f32::Vec2;
use rand::random;
use std::{slice, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorBufferInfo, DescriptorImageInfo,
        DescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, ComputePipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
};

pub struct FractalComputePipeline {
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    palette: Subbuffer<[[f32; 4]]>,
    palette_size: i32,
    end_color: [f32; 4],
}

impl FractalComputePipeline {
    pub fn new(
        queue: &Arc<Queue>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    ) -> FractalComputePipeline {
        // Initial colors.
        let colors = vec![
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ];
        let palette_size = colors.len() as i32;
        let palette = Buffer::from_iter(
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
            colors,
        )
        .unwrap();
        let end_color = [0.0; 4];

        let pipeline = {
            let device = queue.device();
            let cs = unsafe { cs::load(device) }
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

        FractalComputePipeline {
            queue: queue.clone(),
            pipeline,
            memory_allocator: memory_allocator.clone(),
            command_buffer_allocator: command_buffer_allocator.clone(),
            descriptor_set_allocator: descriptor_set_allocator.clone(),
            palette,
            palette_size,
            end_color,
        }
    }

    /// Randomizes our color palette.
    pub fn randomize_palette(&mut self) {
        let mut colors = vec![];
        for _ in 0..self.palette_size {
            let r = random::<f32>();
            let g = random::<f32>();
            let b = random::<f32>();
            let a = random::<f32>();
            colors.push([r, g, b, a]);
        }
        self.palette = Buffer::from_iter(
            &self.memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            colors,
        )
        .unwrap();
    }

    pub fn compute(
        &self,
        image_view: Arc<ImageView>,
        c: Vec2,
        scale: Vec2,
        translation: Vec2,
        max_iters: u32,
        is_julia: bool,
    ) -> Box<dyn GpuFuture> {
        // Resize image if needed.
        let image_extent = image_view.image().extent();
        let layout = &self.pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            &self.descriptor_set_allocator,
            layout,
            &[
                WriteDescriptorSet::image(
                    0,
                    &DescriptorImageInfo {
                        image_view: Some(&image_view),
                        ..Default::default()
                    },
                ),
                WriteDescriptorSet::buffer(
                    1,
                    &DescriptorBufferInfo {
                        buffer: Some(self.palette.buffer()),
                        ..Default::default()
                    },
                ),
            ],
            &[],
        )
        .unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let push_constants = cs::PushConstants {
            end_color: self.end_color,
            c: c.into(),
            scale: scale.into(),
            translation: translation.into(),
            palette_size: self.palette_size,
            max_iters: max_iters as i32,
            is_julia: is_julia as u32,
        };

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .unwrap();
        unsafe { builder.dispatch([image_extent[0] / 8, image_extent[1] / 8, 1]) }.unwrap();

        let command_buffer = builder.build().unwrap();
        let finished = command_buffer.execute(self.queue.clone()).unwrap();
        finished.then_signal_fence_and_flush().unwrap().boxed()
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            // Image to which we'll write our fractal
            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

            // Our palette as a dynamic buffer
            layout(set = 0, binding = 1) buffer Palette {
                vec4 data[];
            } palette;

            // Our variable inputs as push constants
            layout(push_constant) uniform PushConstants {
                vec4 end_color;
                vec2 c;
                vec2 scale;
                vec2 translation;
                int palette_size;
                int max_iters;
                bool is_julia;
            } push_constants;

            // Gets smooth color between current color (determined by iterations) and the next 
            // color in the palette by linearly interpolating the colors based on: 
            // https://linas.org/art-gallery/escape/smooth.html
            vec4 get_color(
                int palette_size,
                vec4 end_color,
                int i,
                int max_iters,
                float len_z
            ) {
                if (i < max_iters) {
                    float iters_float = float(i) + 1.0 - log(log(len_z)) / log(2.0f);
                    float iters_floor = floor(iters_float);
                    float remainder = iters_float - iters_floor;
                    vec4 color_start = palette.data[int(iters_floor) % push_constants.palette_size];
                    vec4 color_end = palette.data[(int(iters_floor) + 1) % push_constants.palette_size];
                    return mix(color_start, color_end, remainder);
                }
                return end_color;
            }

            void main() {
                // Scale image pixels to range
                vec2 dims = vec2(imageSize(img));
                float ar = dims.x / dims.y;
                float x_over_width = (gl_GlobalInvocationID.x / dims.x);
                float y_over_height = (gl_GlobalInvocationID.y / dims.y);
                float x0 = ar * (push_constants.translation.x + (x_over_width - 0.5) * push_constants.scale.x);
                float y0 = push_constants.translation.y + (y_over_height - 0.5) * push_constants.scale.y;

                // Julia is like mandelbrot, but instead changing the constant `c` will change the 
                // shape you'll see. Thus we want to bind the c to mouse position.
                // With mandelbrot, c = scaled xy position of the image. Z starts from zero.
                // With julia, c = any value between the interesting range (-2.0 - 2.0), 
                // Z = scaled xy position of the image.
                vec2 c;
                vec2 z;
                if (push_constants.is_julia) {
                    c = push_constants.c;
                    z = vec2(x0, y0);
                } else {
                    c = vec2(x0, y0);
                    z = vec2(0.0, 0.0);
                }

                // Escape time algorithm:
                // https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
                // It's an iterative algorithm where the bailout point (number of iterations) will 
                // determine the color we choose from the palette.
                int i;
                float len_z;
                for (i = 0; i < push_constants.max_iters; i += 1) {
                    z = vec2(
                        z.x * z.x - z.y * z.y + c.x,
                        z.y * z.x + z.x * z.y + c.y
                    );

                    len_z = length(z);
                    // Using 8.0 for bailout limit give a little nicer colors with smooth colors
                    // 2.0 is enough to 'determine' an escape will happen.
                    if (len_z > 8.0) {
                        break;
                    }
                }

                vec4 write_color = get_color(
                    push_constants.palette_size,
                    push_constants.end_color,
                    i,
                    push_constants.max_iters,
                    len_z
                );
                imageStore(img, ivec2(gl_GlobalInvocationID.xy), write_color);
            }
        ",
    }
}
