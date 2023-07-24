fn main() {
    wasm_logger::init(wasm_logger::Config::new(log::Level::Trace));
    yew::Renderer::<candle_wasm_example_llama2::App>::new().render();
}
