#![allow(dead_code)]
use wasm_bindgen::prelude::*;

mod app;
mod audio;
mod model;

#[wasm_bindgen]
pub fn run_app() -> Result<(), JsValue> {
    wasm_logger::init(wasm_logger::Config::new(log::Level::Trace));
    yew::Renderer::<app::App>::new().render();

    Ok(())
}
