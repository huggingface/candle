use crate::pixels_draw_pipeline::PixelsDrawPipeline;
use std::sync::Arc;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
    format::Format,
    image::view::ImageView,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};

/// A render pass which places an incoming image over frame filling it.
pub struct RenderPassPlaceOverFrame {
    gfx_queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pixels_draw_pipeline: PixelsDrawPipeline,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

impl RenderPassPlaceOverFrame {
    pub fn new(
        gfx_queue: &Arc<Queue>,
        command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        output_format: Format,
        swapchain_image_views: &[Arc<ImageView>],
    ) -> RenderPassPlaceOverFrame {
        let render_pass = vulkano::single_pass_renderpass!(
            gfx_queue.device(),
            attachments: {
                color: {
                    format: output_format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();
        let subpass = Subpass::new(&render_pass, 0).unwrap();
        let pixels_draw_pipeline = PixelsDrawPipeline::new(
            gfx_queue,
            &subpass,
            command_buffer_allocator,
            descriptor_set_allocator,
        );
        let framebuffers = create_framebuffers(swapchain_image_views, &render_pass);

        RenderPassPlaceOverFrame {
            gfx_queue: gfx_queue.clone(),
            render_pass,
            pixels_draw_pipeline,
            command_buffer_allocator: command_buffer_allocator.clone(),
            framebuffers,
        }
    }

    /// Places the view exactly over the target swapchain image. The texture draw pipeline uses a
    /// quad onto which it places the view.
    pub fn render<F>(
        &self,
        before_future: F,
        view: Arc<ImageView>,
        target: Arc<ImageView>,
        image_index: u32,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        // Get dimensions.
        let img_dims: [u32; 2] = target.image().extent()[0..2].try_into().unwrap();

        // Create primary command buffer builder.
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Begin render pass.
        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0; 4].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..Default::default()
                },
            )
            .unwrap();

        // Create secondary command buffer from texture pipeline & send draw commands.
        let cb = self.pixels_draw_pipeline.draw(img_dims, view);

        // Execute above commands (subpass).
        command_buffer_builder.execute_commands(cb).unwrap();

        // End render pass.
        command_buffer_builder
            .end_render_pass(Default::default())
            .unwrap();

        // Build command buffer.
        let command_buffer = command_buffer_builder.build().unwrap();

        // Execute primary command buffer.
        let after_future = before_future
            .then_execute(self.gfx_queue.clone(), command_buffer)
            .unwrap();

        after_future.boxed()
    }

    pub fn recreate_framebuffers(&mut self, swapchain_image_views: &[Arc<ImageView>]) {
        self.framebuffers = create_framebuffers(swapchain_image_views, &self.render_pass);
    }
}

fn create_framebuffers(
    swapchain_image_views: &[Arc<ImageView>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    swapchain_image_views
        .iter()
        .map(|swapchain_image_view| {
            Framebuffer::new(
                render_pass,
                &FramebufferCreateInfo {
                    attachments: &[swapchain_image_view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
