// Modified from https://github.com/bevyengine/bevy/tree/main/crates/bevy_window, to fit Vulkano.
// Their licences: https://github.com/bevyengine/bevy/blob/main/LICENSE-MIT
// https://github.com/bevyengine/bevy/blob/main/LICENSE-APACHE

use crate::{context::VulkanoContext, renderer::VulkanoWindowRenderer};
use foldhash::HashMap;
use std::collections::hash_map::{Iter, IterMut};
use vulkano::swapchain::{PresentMode, SwapchainCreateInfo};
use winit::{
    dpi::LogicalSize,
    event_loop::ActiveEventLoop,
    window::{CursorGrabMode, WindowId},
};

/// A struct organizing windows and their corresponding renderers. This makes it easy to handle
/// multiple windows.
///
/// ## Examples
///
/// ```
/// use vulkano_util::{
///     context::{VulkanoConfig, VulkanoContext},
///     window::VulkanoWindows,
/// };
///
/// # let event_loop = return;
/// let context = VulkanoContext::new(VulkanoConfig::default());
/// let mut vulkano_windows = VulkanoWindows::default();
/// let _id1 = vulkano_windows.create_window(event_loop, &context, &Default::default(), |_| {});
/// let _id2 = vulkano_windows.create_window(event_loop, &context, &Default::default(), |_| {});
///
/// // You should now have two windows.
/// ```
#[derive(Default)]
pub struct VulkanoWindows {
    windows: HashMap<WindowId, VulkanoWindowRenderer>,
    primary: Option<WindowId>,
}

impl VulkanoWindows {
    /// Creates a winit window with [`VulkanoWindowRenderer`] based on the given
    /// [`WindowDescriptor`] input and swapchain creation modifications.
    pub fn create_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        vulkano_context: &VulkanoContext,
        window_descriptor: &WindowDescriptor,
        swapchain_create_info_modify: fn(&mut SwapchainCreateInfo<'_>),
    ) -> WindowId {
        let mut winit_window_attributes = winit::window::Window::default_attributes();

        winit_window_attributes = match window_descriptor.mode {
            WindowMode::BorderlessFullscreen => winit_window_attributes.with_fullscreen(Some(
                winit::window::Fullscreen::Borderless(event_loop.primary_monitor()),
            )),
            WindowMode::Fullscreen => {
                winit_window_attributes.with_fullscreen(Some(winit::window::Fullscreen::Exclusive(
                    get_best_videomode(&event_loop.primary_monitor().unwrap()),
                )))
            }
            WindowMode::SizedFullscreen => winit_window_attributes.with_fullscreen(Some(
                winit::window::Fullscreen::Exclusive(get_fitting_videomode(
                    &event_loop.primary_monitor().unwrap(),
                    window_descriptor.width as u32,
                    window_descriptor.height as u32,
                )),
            )),
            _ => {
                let WindowDescriptor {
                    width,
                    height,
                    position,
                    scale_factor_override,
                    ..
                } = window_descriptor;

                if let Some(position) = position {
                    if let Some(sf) = scale_factor_override {
                        winit_window_attributes = winit_window_attributes.with_position(
                            winit::dpi::LogicalPosition::new(
                                position[0] as f64,
                                position[1] as f64,
                            )
                            .to_physical::<f64>(*sf),
                        );
                    } else {
                        winit_window_attributes = winit_window_attributes.with_position(
                            winit::dpi::LogicalPosition::new(
                                position[0] as f64,
                                position[1] as f64,
                            ),
                        );
                    }
                }

                if let Some(sf) = scale_factor_override {
                    winit_window_attributes
                        .with_inner_size(LogicalSize::new(*width, *height).to_physical::<f64>(*sf))
                } else {
                    winit_window_attributes.with_inner_size(LogicalSize::new(*width, *height))
                }
            }
            .with_resizable(window_descriptor.resizable)
            .with_decorations(window_descriptor.decorations)
            .with_transparent(window_descriptor.transparent),
        };

        let constraints = window_descriptor.resize_constraints.check_constraints();
        let min_inner_size = LogicalSize {
            width: constraints.min_width,
            height: constraints.min_height,
        };
        let max_inner_size = LogicalSize {
            width: constraints.max_width,
            height: constraints.max_height,
        };

        let winit_window_attributes =
            if constraints.max_width.is_finite() && constraints.max_height.is_finite() {
                winit_window_attributes
                    .with_min_inner_size(min_inner_size)
                    .with_max_inner_size(max_inner_size)
            } else {
                winit_window_attributes.with_min_inner_size(min_inner_size)
            };

        #[allow(unused_mut)]
        let mut winit_window_attributes =
            winit_window_attributes.with_title(&window_descriptor.title);

        let winit_window = event_loop.create_window(winit_window_attributes).unwrap();

        if window_descriptor.cursor_locked {
            match winit_window.set_cursor_grab(CursorGrabMode::Confined) {
                Ok(_) => {}
                Err(winit::error::ExternalError::NotSupported(_)) => {}
                Err(err) => panic!("{:?}", err),
            }
        }

        winit_window.set_cursor_visible(window_descriptor.cursor_visible);

        let id = winit_window.id();
        if self.primary.is_none() {
            self.primary = Some(id);
        }

        self.windows.insert(
            id,
            VulkanoWindowRenderer::new(
                vulkano_context,
                winit_window,
                window_descriptor,
                swapchain_create_info_modify,
            ),
        );

        id
    }

    /// Get a mutable reference to the primary window's renderer.
    #[inline]
    pub fn get_primary_renderer_mut(&mut self) -> Option<&mut VulkanoWindowRenderer> {
        if let Some(primary) = self.primary {
            self.get_renderer_mut(primary)
        } else {
            None
        }
    }

    /// Get a reference to the primary window's renderer.
    #[inline]
    pub fn get_primary_renderer(&self) -> Option<&VulkanoWindowRenderer> {
        if let Some(primary) = self.primary {
            self.get_renderer(primary)
        } else {
            None
        }
    }

    /// Get a reference to the primary winit window.
    #[inline]
    pub fn get_primary_window(&self) -> Option<&winit::window::Window> {
        if let Some(primary) = self.primary {
            self.get_window(primary)
        } else {
            None
        }
    }

    /// Get a mutable reference to the renderer by winit window id.
    #[inline]
    pub fn get_renderer_mut(&mut self, id: WindowId) -> Option<&mut VulkanoWindowRenderer> {
        self.windows.get_mut(&id)
    }

    /// Get a reference to the renderer by winit window id.
    #[inline]
    pub fn get_renderer(&self, id: WindowId) -> Option<&VulkanoWindowRenderer> {
        self.windows.get(&id)
    }

    /// Get a reference to the winit window by winit window id.
    #[inline]
    pub fn get_window(&self, id: WindowId) -> Option<&winit::window::Window> {
        self.windows.get(&id).map(|v_window| v_window.window())
    }

    /// Return primary window id.
    #[inline]
    pub fn primary_window_id(&self) -> Option<WindowId> {
        self.primary
    }

    /// Remove renderer by window id.
    #[inline]
    pub fn remove_renderer(&mut self, id: WindowId) {
        self.windows.remove(&id);
        if let Some(primary) = self.primary {
            if primary == id {
                self.primary = None;
            }
        }
    }

    /// Return iterator over window renderers.
    #[inline]
    pub fn iter(&self) -> Iter<'_, WindowId, VulkanoWindowRenderer> {
        self.windows.iter()
    }

    /// Return iterator over mutable window renderers.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, WindowId, VulkanoWindowRenderer> {
        self.windows.iter_mut()
    }
}

fn get_fitting_videomode(
    monitor: &winit::monitor::MonitorHandle,
    width: u32,
    height: u32,
) -> winit::monitor::VideoModeHandle {
    let mut modes = monitor.video_modes().collect::<Vec<_>>();

    fn abs_diff(a: u32, b: u32) -> u32 {
        if a > b {
            return a - b;
        }
        b - a
    }

    modes.sort_by(|a, b| {
        use std::cmp::Ordering::*;
        match abs_diff(a.size().width, width).cmp(&abs_diff(b.size().width, width)) {
            Equal => {
                match abs_diff(a.size().height, height).cmp(&abs_diff(b.size().height, height)) {
                    Equal => b
                        .refresh_rate_millihertz()
                        .cmp(&a.refresh_rate_millihertz()),
                    default => default,
                }
            }
            default => default,
        }
    });

    modes.first().unwrap().clone()
}

fn get_best_videomode(monitor: &winit::monitor::MonitorHandle) -> winit::monitor::VideoModeHandle {
    let mut modes = monitor.video_modes().collect::<Vec<_>>();
    modes.sort_by(|a, b| {
        use std::cmp::Ordering::*;
        match b.size().width.cmp(&a.size().width) {
            Equal => match b.size().height.cmp(&a.size().height) {
                Equal => b
                    .refresh_rate_millihertz()
                    .cmp(&a.refresh_rate_millihertz()),
                default => default,
            },
            default => default,
        }
    });

    modes.first().unwrap().clone()
}

/// Defines the way a window is displayed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowMode {
    /// Creates a window that uses the given size.
    Windowed,
    /// Creates a borderless window that uses the full size of the screen.
    BorderlessFullscreen,
    /// Creates a fullscreen window that will render at desktop resolution.
    ///
    /// The app will use the closest supported size from the given size and scale it to fit the
    /// screen.
    SizedFullscreen,
    /// Creates a fullscreen window that uses the maximum supported size.
    Fullscreen,
}

/// Describes the information needed for creating a window.
#[derive(Debug, Clone)]
pub struct WindowDescriptor {
    /// The requested logical width of the window's client area.
    ///
    /// May vary from the physical width due to different pixel density on different monitors.
    pub width: f32,
    /// The requested logical height of the window's client area.
    ///
    /// May vary from the physical height due to different pixel density on different monitors.
    pub height: f32,
    /// The position on the screen that the window will be centered at.
    ///
    /// If set to `None`, some platform-specific position will be chosen.
    pub position: Option<[f32; 2]>,
    /// Sets minimum and maximum resize limits.
    pub resize_constraints: WindowResizeConstraints,
    /// Overrides the window's ratio of physical pixels to logical pixels.
    ///
    /// If there are some scaling problems on X11 try to set this option to `Some(1.0)`.
    pub scale_factor_override: Option<f64>,
    /// Sets the title that displays on the window top bar, on the system task bar and other OS
    /// specific places.
    pub title: String,
    /// The window's [`PresentMode`].
    ///
    /// Used to select whether or not VSync is used
    pub present_mode: PresentMode,
    /// Sets whether the window is resizable.
    pub resizable: bool,
    /// Sets whether the window should have borders and bars.
    pub decorations: bool,
    /// Sets whether the cursor is visible when the window has focus.
    pub cursor_visible: bool,
    /// Sets whether the window locks the cursor inside its borders when the window has focus.
    pub cursor_locked: bool,
    /// Sets the [`WindowMode`].
    pub mode: WindowMode,
    /// Sets whether the background of the window should be transparent.
    pub transparent: bool,
}

impl Default for WindowDescriptor {
    #[inline]
    fn default() -> Self {
        WindowDescriptor {
            title: "Vulkano App".to_string(),
            width: 1280.,
            height: 720.,
            position: None,
            resize_constraints: WindowResizeConstraints::default(),
            scale_factor_override: None,
            present_mode: PresentMode::Fifo,
            resizable: true,
            decorations: true,
            cursor_locked: false,
            cursor_visible: true,
            mode: WindowMode::Windowed,
            transparent: false,
        }
    }
}

/// The size limits on a window.
///
/// These values are measured in logical pixels, so the user's
/// scale factor does affect the size limits on the window.
/// Please note that if the window is resizable, then when the window is
/// maximized it may have a size outside of these limits. The functionality
/// required to disable maximizing is not yet exposed by winit.
#[derive(Debug, Clone, Copy)]
pub struct WindowResizeConstraints {
    pub min_width: f32,
    pub min_height: f32,
    pub max_width: f32,
    pub max_height: f32,
}

impl Default for WindowResizeConstraints {
    #[inline]
    fn default() -> Self {
        Self {
            min_width: 180.,
            min_height: 120.,
            max_width: f32::INFINITY,
            max_height: f32::INFINITY,
        }
    }
}

impl WindowResizeConstraints {
    #[must_use]
    pub fn check_constraints(&self) -> Self {
        let &Self {
            mut min_width,
            mut min_height,
            mut max_width,
            mut max_height,
        } = self;
        min_width = min_width.max(1.);
        min_height = min_height.max(1.);
        if max_width < min_width {
            println!(
                "The given maximum width {} is smaller than the minimum width {}",
                max_width, min_width
            );
            max_width = min_width;
        }
        if max_height < min_height {
            println!(
                "The given maximum height {} is smaller than the minimum height {}",
                max_height, min_height
            );
            max_height = min_height;
        }
        WindowResizeConstraints {
            min_width,
            min_height,
            max_width,
            max_height,
        }
    }
}
