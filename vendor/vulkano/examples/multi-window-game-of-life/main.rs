// A multi windowed game of life application. You could use this to learn:
//
// - how to handle multiple window inputs,
// - how to draw on a canvas,
// - how to organize compute shader with graphics,
// - how to do a cellular automata simulation using compute shaders.
//
// The possibilities are limitless. ;)

use game_of_life::GameOfLifeComputePipeline;
use glam::{f32::Vec2, IVec2};
use render_pass::RenderPassPlaceOverFrame;
use std::{
    collections::HashMap,
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::VulkanoWindowRenderer,
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler,
    event::{MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

mod game_of_life;
mod pixels_draw;
mod render_pass;

const WINDOW_WIDTH: f32 = 1024.0;
const WINDOW_HEIGHT: f32 = 1024.0;
const WINDOW2_WIDTH: f32 = 512.0;
const WINDOW2_HEIGHT: f32 = 512.0;
const SCALING: f32 = 2.0;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    println!("Welcome to Vulkano Game of Life\nUse the mouse to draw life on the grid(s)\n");

    event_loop.run_app(&mut app)
}

struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    rcxs: HashMap<WindowId, RenderContext>,
    time: Instant,
    cursor_pos: Vec2,
}

struct RenderContext {
    compute_pipeline: GameOfLifeComputePipeline,
    place_over_frame: RenderPassPlaceOverFrame,
    life_color: [f32; 4],
    dead_color: [f32; 4],
    mouse_is_pressed: bool,
}

impl App {
    fn new(_event_loop: &EventLoop<()>) -> Self {
        let context = VulkanoContext::new(VulkanoConfig::default());
        let windows = VulkanoWindows::default();
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device(),
            &Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device(),
            &StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));

        // Time & inputs...
        let time = Instant::now();
        let cursor_pos = Vec2::ZERO;

        App {
            context,
            windows,
            descriptor_set_allocator,
            command_buffer_allocator,
            rcxs: HashMap::new(),
            time,
            cursor_pos,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create windows & pipelines.
        let id1 = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                width: WINDOW_WIDTH,
                height: WINDOW_HEIGHT,
                title: "Game of Life Primary".to_string(),
                ..Default::default()
            },
            |_| {},
        );
        let id2 = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                width: WINDOW2_WIDTH,
                height: WINDOW2_HEIGHT,
                title: "Game of Life Secondary".to_string(),
                ..Default::default()
            },
            |_| {},
        );
        let gfx_queue = self.context.graphics_queue();
        self.rcxs.insert(
            id1,
            RenderContext {
                compute_pipeline: GameOfLifeComputePipeline::new(
                    self,
                    gfx_queue,
                    [
                        (WINDOW_WIDTH / SCALING) as u32,
                        (WINDOW_HEIGHT / SCALING) as u32,
                    ],
                ),
                place_over_frame: RenderPassPlaceOverFrame::new(self, gfx_queue, id1),
                life_color: [1.0, 0.0, 0.0, 1.0],
                dead_color: [0.0; 4],
                mouse_is_pressed: false,
            },
        );
        self.rcxs.insert(
            id2,
            RenderContext {
                compute_pipeline: GameOfLifeComputePipeline::new(
                    self,
                    gfx_queue,
                    [
                        (WINDOW2_WIDTH / SCALING) as u32,
                        (WINDOW2_HEIGHT / SCALING) as u32,
                    ],
                ),
                place_over_frame: RenderPassPlaceOverFrame::new(self, gfx_queue, id2),
                life_color: [0.0, 0.0, 0.0, 1.0],
                dead_color: [1.0; 4],
                mouse_is_pressed: false,
            },
        );
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if window_id == self.windows.primary_window_id().unwrap() {
                    event_loop.exit();
                } else {
                    // Destroy window by removing its renderer.
                    self.windows.remove_renderer(window_id);
                    self.rcxs.remove(&window_id);
                }
            }
            // Resize window and its images.
            WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. } => {
                let window_renderer = self.windows.get_renderer_mut(window_id).unwrap();
                window_renderer.resize();
            }
            // Handle mouse position events.
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = Vec2::from_array(position.into());
            }
            // Handle mouse button events.
            WindowEvent::MouseInput { state, button, .. } => {
                let rcx = self.rcxs.get_mut(&window_id).unwrap();

                if button == MouseButton::Left {
                    rcx.mouse_is_pressed = state.is_pressed();
                }
            }
            WindowEvent::RedrawRequested => {
                let Some(window_renderer) = self.windows.get_renderer_mut(window_id) else {
                    return;
                };
                let rcx = self.rcxs.get_mut(&window_id).unwrap();
                let window_size = window_renderer.window().inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                // Draw life on windows if mouse is down.
                draw_life(window_renderer, rcx, self.cursor_pos);

                // Compute life & render 60fps.
                compute_then_render(window_renderer, rcx);
                self.time = Instant::now();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        for (_, renderer) in self.windows.iter() {
            renderer.window().request_redraw();
        }
    }
}

fn draw_life(
    window_renderer: &mut VulkanoWindowRenderer,
    rcx: &mut RenderContext,
    cursor_pos: Vec2,
) {
    if rcx.mouse_is_pressed {
        let window_size = window_renderer.window_size();
        let mut normalized_pos = Vec2::new(
            (cursor_pos.x / window_size[0]).clamp(0.0, 1.0),
            (cursor_pos.y / window_size[1]).clamp(0.0, 1.0),
        );

        // Flip y.
        normalized_pos.y = 1.0 - normalized_pos.y;
        let image_extent = rcx.compute_pipeline.color_image().image().extent();
        rcx.compute_pipeline.draw_life(IVec2::new(
            (image_extent[0] as f32 * normalized_pos.x) as i32,
            (image_extent[1] as f32 * normalized_pos.y) as i32,
        ));
    }
}

/// Compute game of life, then display result on target image.
fn compute_then_render(window_renderer: &mut VulkanoWindowRenderer, rcx: &mut RenderContext) {
    // Start the frame.
    let before_pipeline_future =
        match window_renderer.acquire(Some(Duration::from_millis(1000)), |swapchain_image_views| {
            rcx.place_over_frame
                .recreate_framebuffers(swapchain_image_views)
        }) {
            Err(e) => {
                println!("{e}");
                return;
            }
            Ok(future) => future,
        };

    // Compute.
    let after_compute =
        rcx.compute_pipeline
            .compute(before_pipeline_future, rcx.life_color, rcx.dead_color);

    // Render.
    let color_image = rcx.compute_pipeline.color_image();
    let target_image = window_renderer.swapchain_image_view();

    let after_render = rcx.place_over_frame.render(
        after_compute,
        color_image,
        target_image,
        window_renderer.image_index(),
    );

    // Finish the frame. Wait for the future so resources are not in use when we render.
    window_renderer.present(after_render, true);
}
