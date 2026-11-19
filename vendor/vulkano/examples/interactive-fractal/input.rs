use glam::f32::Vec2;
use vulkano_util::window::WindowDescriptor;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{Key, NamedKey},
};

/// Just a very simple input state (mappings). Winit only has `Pressed` and `Released` events, thus
/// continuous movement needs toggles. Panning is one of those things where continuous movement
/// feels better.
pub struct InputState {
    pub window_size: [f32; 2],
    pub pan_up: bool,
    pub pan_down: bool,
    pub pan_right: bool,
    pub pan_left: bool,
    pub increase_iterations: bool,
    pub decrease_iterations: bool,
    pub randomize_palette: bool,
    pub toggle_full_screen: bool,
    pub toggle_julia: bool,
    pub toggle_c: bool,
    pub should_quit: bool,
    pub scroll_delta: f32,
    pub mouse_pos: Vec2,
}

impl InputState {
    pub fn new() -> InputState {
        InputState {
            window_size: [
                WindowDescriptor::default().width,
                WindowDescriptor::default().height,
            ],
            pan_up: false,
            pan_down: false,
            pan_right: false,
            pan_left: false,
            increase_iterations: false,
            decrease_iterations: false,
            randomize_palette: false,
            toggle_full_screen: false,
            toggle_julia: false,
            toggle_c: false,
            should_quit: false,
            scroll_delta: 0.0,
            mouse_pos: Vec2::new(0.0, 0.0),
        }
    }

    pub fn normalized_mouse_pos(&self) -> Vec2 {
        Vec2::new(
            (self.mouse_pos.x / self.window_size[0]).clamp(0.0, 1.0),
            (self.mouse_pos.y / self.window_size[1]).clamp(0.0, 1.0),
        )
    }

    /// Resets values that should be reset. All incremental mappings and toggles should be reset.
    pub fn reset(&mut self) {
        *self = InputState {
            scroll_delta: 0.0,
            toggle_full_screen: false,
            toggle_julia: false,
            toggle_c: false,
            randomize_palette: false,
            increase_iterations: false,
            decrease_iterations: false,
            ..*self
        }
    }

    pub fn handle_input(&mut self, window_size: PhysicalSize<u32>, event: &WindowEvent) {
        self.window_size = window_size.into();

        match event {
            WindowEvent::KeyboardInput { event, .. } => self.on_keyboard_event(event),
            WindowEvent::MouseInput { state, button, .. } => {
                self.on_mouse_click_event(*state, *button)
            }
            WindowEvent::CursorMoved { position, .. } => self.on_cursor_moved_event(position),
            WindowEvent::MouseWheel { delta, .. } => self.on_mouse_wheel_event(delta),
            _ => {}
        }
    }

    /// Matches keyboard events to our defined inputs.
    fn on_keyboard_event(&mut self, event: &KeyEvent) {
        match event.logical_key.as_ref() {
            Key::Named(NamedKey::Escape) => self.should_quit = event.state.is_pressed(),
            Key::Character("w") => self.pan_up = event.state.is_pressed(),
            Key::Character("a") => self.pan_left = event.state.is_pressed(),
            Key::Character("s") => self.pan_down = event.state.is_pressed(),
            Key::Character("d") => self.pan_right = event.state.is_pressed(),
            Key::Character("f") => self.toggle_full_screen = event.state.is_pressed(),
            Key::Named(NamedKey::Enter) => self.randomize_palette = event.state.is_pressed(),
            Key::Character("=") => self.increase_iterations = event.state.is_pressed(),
            Key::Character("-") => self.decrease_iterations = event.state.is_pressed(),
            Key::Named(NamedKey::Space) => self.toggle_julia = event.state.is_pressed(),
            _ => {}
        }
    }

    /// Updates mouse scroll delta.
    fn on_mouse_wheel_event(&mut self, delta: &MouseScrollDelta) {
        let change = match delta {
            MouseScrollDelta::LineDelta(_x, y) => *y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
        };
        self.scroll_delta += change;
    }

    /// Update mouse position
    fn on_cursor_moved_event(&mut self, pos: &PhysicalPosition<f64>) {
        self.mouse_pos = Vec2::new(pos.x as f32, pos.y as f32);
    }

    /// Update toggle julia state (if right mouse is clicked)
    fn on_mouse_click_event(&mut self, state: ElementState, mouse_btn: MouseButton) {
        if mouse_btn == MouseButton::Right {
            self.toggle_c = state.is_pressed();
        }
    }
}
