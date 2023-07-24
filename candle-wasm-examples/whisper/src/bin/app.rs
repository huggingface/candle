fn main() {
    wasm_logger::init(wasm_logger::Config::new(log::Level::Trace));
    yew::Renderer::<candle_wasm_example_whisper::App>::new().render();
}
