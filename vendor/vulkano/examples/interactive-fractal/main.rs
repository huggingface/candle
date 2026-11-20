// This is an example demonstrating an application with some more non-trivial functionality.
// It should get you more up to speed with how you can use Vulkano.
//
// It contains:
//
// - A compute pipeline to calculate Mandelbrot and Julia fractals writing them to an image.
// - A graphics pipeline to draw the fractal image over a quad that covers the whole screen.
// - A renderpass rendering that image on the swapchain image.
// - An organized renderer with functionality good enough to copy to other projects.
// - A simple `FractalApp` to handle runtime state.
// - A simple `InputState` to interact with the application.

use fractal_compute_pipeline::FractalComputePipeline;
use glam::Vec2;
use input::InputState;
use place_over_frame::RenderPassPlaceOverFrame;
use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    image::ImageUsage,
    swapchain::PresentMode,
    sync::GpuFuture,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::{VulkanoWindowRenderer, DEFAULT_IMAGE_FORMAT},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Fullscreen, WindowId},
};

mod fractal_compute_pipeline;
mod input;
mod pixels_draw_pipeline;
mod place_over_frame;

const MAX_ITERS_INIT: u32 = 200;
const MOVE_SPEED: f32 = 0.5;

fn main() -> Result<(), impl Error> {
    // Create the event loop.
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    println!(
        "\
Usage:
    WASD: Pan view
    Scroll: Zoom in/out
    Space: Toggle between Mandelbrot and Julia
    Enter: Randomize color palette
    Equals/Minus: Increase/Decrease max iterations
    F: Toggle full-screen
    Right mouse: Stop movement in Julia (mouse position determines c)
    Esc: Quit\
        ",
    );

    event_loop.run_app(&mut app)
}

struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    /// Pipeline that computes Mandelbrot & Julia fractals and writes them to an image.
    fractal_pipeline: FractalComputePipeline,
    /// Our render pipeline (pass).
    place_over_frame: RenderPassPlaceOverFrame,
    /// Toggle that flips between Julia and Mandelbrot.
    is_julia: bool,
    /// Toggle that stops the movement on Julia.
    is_c_paused: bool,
    /// C is a constant input to Julia escape time algorithm (mouse position).
    c: Vec2,
    /// Our zoom level.
    scale: Vec2,
    /// Our translation on the complex plane.
    translation: Vec2,
    /// How long the escape time algorithm should run (higher = less performance, more accurate
    /// image).
    max_iters: u32,
    /// Time tracking, useful for frame independent movement.
    time: Instant,
    dt: f32,
    dt_sum: f32,
    frame_count: f32,
    avg_fps: f32,
    /// Input state to handle mouse positions, continuous movement etc.
    input_state: InputState,
    render_target_id: usize,
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

        App {
            context,
            windows,
            descriptor_set_allocator,
            command_buffer_allocator,
            rcx: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let _id = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                title: "Fractal".to_string(),
                present_mode: PresentMode::Fifo,
                ..Default::default()
            },
            |_| {},
        );

        // Add our render target image onto which we'll be rendering our fractals.
        let render_target_id = 0;
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();

        // Make sure the image usage is correct (based on your pipeline).
        window_renderer.add_additional_image_view(
            render_target_id,
            DEFAULT_IMAGE_FORMAT,
            ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
        );

        let gfx_queue = self.context.graphics_queue();

        self.rcx = Some(RenderContext {
            render_target_id,
            fractal_pipeline: FractalComputePipeline::new(
                gfx_queue,
                self.context.memory_allocator(),
                &self.command_buffer_allocator,
                &self.descriptor_set_allocator,
            ),
            place_over_frame: RenderPassPlaceOverFrame::new(
                gfx_queue,
                &self.command_buffer_allocator,
                &self.descriptor_set_allocator,
                window_renderer.swapchain_format(),
                window_renderer.swapchain_image_views(),
            ),
            is_julia: false,
            is_c_paused: false,
            c: Vec2::new(0.0, 0.0),
            scale: Vec2::new(4.0, 4.0),
            translation: Vec2::new(0.0, 0.0),
            max_iters: MAX_ITERS_INIT,
            time: Instant::now(),
            dt: 0.0,
            dt_sum: 0.0,
            frame_count: 0.0,
            avg_fps: 0.0,
            input_state: InputState::new(),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let renderer = self.windows.get_primary_renderer_mut().unwrap();
        let rcx = self.rcx.as_mut().unwrap();
        let window_size = renderer.window().inner_size();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. } => {
                renderer.resize();
            }
            WindowEvent::RedrawRequested => {
                // Tasks for redrawing:
                // 1. Update state based on events
                // 2. Compute & Render
                // 3. Reset input state
                // 4. Update time & title

                // Skip this frame when minimized.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                rcx.update_state_after_inputs(renderer);

                // Start the frame.
                let before_pipeline_future = match renderer.acquire(
                    Some(Duration::from_millis(1000)),
                    |swapchain_image_views| {
                        rcx.place_over_frame
                            .recreate_framebuffers(swapchain_image_views)
                    },
                ) {
                    Err(e) => {
                        println!("{e}");
                        return;
                    }
                    Ok(future) => future,
                };

                // Retrieve the target image.
                let image = renderer.get_additional_image_view(rcx.render_target_id);

                // Compute our fractal (writes to target image). Join future with
                // `before_pipeline_future`.
                let after_compute = rcx
                    .fractal_pipeline
                    .compute(
                        image.clone(),
                        rcx.c,
                        rcx.scale,
                        rcx.translation,
                        rcx.max_iters,
                        rcx.is_julia,
                    )
                    .join(before_pipeline_future);

                // Render the image over the swapchain image, inputting the previous future.
                let after_renderpass_future = rcx.place_over_frame.render(
                    after_compute,
                    image,
                    renderer.swapchain_image_view(),
                    renderer.image_index(),
                );

                // Finish the frame (which presents the view), inputting the last future. Wait for
                // the future so resources are not in use when we render.
                renderer.present(after_renderpass_future, true);

                rcx.input_state.reset();
                rcx.update_time();
                renderer.window().set_title(&format!(
                    "{} fps: {:.2} dt: {:.2}, Max Iterations: {}",
                    if rcx.is_julia { "Julia" } else { "Mandelbrot" },
                    rcx.avg_fps(),
                    rcx.dt(),
                    rcx.max_iters
                ));
            }
            _ => {
                // Pass event for the app to handle our inputs.
                rcx.input_state.handle_input(window_size, &event);
            }
        }

        if rcx.input_state.should_quit {
            event_loop.exit();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        window_renderer.window().request_redraw();
    }
}

impl RenderContext {
    /// Updates app state based on input state.
    fn update_state_after_inputs(&mut self, renderer: &mut VulkanoWindowRenderer) {
        // Zoom in or out.
        if self.input_state.scroll_delta > 0. {
            self.scale /= 1.05;
        } else if self.input_state.scroll_delta < 0. {
            self.scale *= 1.05;
        }

        // Move speed scaled by zoom level.
        let move_speed = MOVE_SPEED * self.dt * self.scale.x;

        // Panning.
        if self.input_state.pan_up {
            self.translation += Vec2::new(0.0, move_speed);
        }
        if self.input_state.pan_down {
            self.translation += Vec2::new(0.0, -move_speed);
        }
        if self.input_state.pan_right {
            self.translation += Vec2::new(move_speed, 0.0);
        }
        if self.input_state.pan_left {
            self.translation += Vec2::new(-move_speed, 0.0);
        }

        // Toggle between Julia and Mandelbrot.
        if self.input_state.toggle_julia {
            self.is_julia = !self.is_julia;
        }

        // Toggle c.
        if self.input_state.toggle_c {
            self.is_c_paused = !self.is_c_paused;
        }

        // Update c.
        if !self.is_c_paused {
            // Scale normalized mouse pos between -1.0 and 1.0.
            let mouse_pos = self.input_state.normalized_mouse_pos() * 2.0 - Vec2::new(1.0, 1.0);
            // Scale by our zoom (scale) level so when zooming in the movement on Julia is not so
            // drastic.
            self.c = mouse_pos * self.scale.x;
        }

        // Update how many iterations we have.
        if self.input_state.increase_iterations {
            self.max_iters += 1;
        }
        if self.input_state.decrease_iterations {
            if self.max_iters as i32 - 1 <= 0 {
                self.max_iters = 0;
            } else {
                self.max_iters -= 1;
            }
        }

        // Randomize our palette.
        if self.input_state.randomize_palette {
            self.fractal_pipeline.randomize_palette();
        }

        // Toggle full-screen.
        if self.input_state.toggle_full_screen {
            let is_full_screen = renderer.window().fullscreen().is_some();
            renderer.window().set_fullscreen(if !is_full_screen {
                Some(Fullscreen::Borderless(renderer.window().current_monitor()))
            } else {
                None
            });
        }
    }

    /// Returns the average FPS.
    fn avg_fps(&self) -> f32 {
        self.avg_fps
    }

    /// Returns the delta time in milliseconds.
    fn dt(&self) -> f32 {
        self.dt * 1000.0
    }

    /// Updates times and dt at the end of each frame.
    fn update_time(&mut self) {
        // Each second, update average fps & reset frame count & dt sum.
        if self.dt_sum > 1.0 {
            self.avg_fps = self.frame_count / self.dt_sum;
            self.frame_count = 0.0;
            self.dt_sum = 0.0;
        }
        self.dt = self.time.elapsed().as_secs_f32();
        self.dt_sum += self.dt;
        self.frame_count += 1.0;
        self.time = Instant::now();
    }
}
