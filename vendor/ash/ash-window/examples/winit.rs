//! Demonstrate interop with beryllium/SDL windows.
//!
//! Sample creates a surface from a window through the
//! platform agnostic window handle trait.
//!
//! On instance extensions platform specific extensions need to be enabled.

use ash::vk;
use std::error::Error;
use winit::{
    dpi::PhysicalSize,
    event::{Event, KeyEvent, WindowEvent},
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::WindowBuilder,
};

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;

    unsafe {
        let entry = ash::Entry::linked();
        let surface_extensions =
            ash_window::enumerate_required_extensions(event_loop.display_handle()?.as_raw())?;
        let app_desc = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 0, 0));
        let instance_desc = vk::InstanceCreateInfo::default()
            .application_info(&app_desc)
            .enabled_extension_names(surface_extensions);

        let instance = entry.create_instance(&instance_desc, None)?;

        let surface_factory = ash_window::SurfaceFactory::new(
            &entry,
            &instance,
            event_loop.display_handle()?.as_raw(),
        )
        .unwrap();

        let window = WindowBuilder::new()
            .with_inner_size(PhysicalSize::<u32>::from((800, 600)))
            .build(&event_loop)?;

        // Load the surface extensions
        let surface_fn = ash::khr::surface::Instance::load(&entry, &instance);
        let mut surface = None;

        let _ = event_loop.run(move |event, elwp| match event {
            winit::event::Event::WindowEvent {
                event:
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key: Key::Named(NamedKey::Escape),
                                ..
                            },
                        ..
                    },
                window_id: _,
            } => {
                elwp.exit();
            }
            Event::LoopExiting => {
                // This will be the last event before the loop terminates.
                // TODO: Winit should issue Suspended consistently on all platforms, matching
                // Resumed and creating a single uniform place to handle surface destruction.
                // https://github.com/rust-windowing/winit/issues/3206
                if let Some(surface) = surface.take() {
                    surface_fn.destroy_surface(surface, None);
                }
            }
            Event::Resumed => {
                // Create a surface from winit window.
                let s = surface_factory
                    .create_surface(window.window_handle().unwrap().as_raw(), None)
                    .unwrap();
                println!("surface: {s:?}");
                assert!(
                    surface.replace(s).is_none(),
                    "Surface must not yet exist when Resumed is called"
                );
            }
            Event::Suspended => {
                let surface = surface
                    .take()
                    .expect("Surface must have been created in Resumed");
                surface_fn.destroy_surface(surface, None);
            }
            _ => {}
        });
        Ok(())
    }
}
