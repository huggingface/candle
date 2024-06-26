fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<candle_wasm_example_llama2::App>::new().render();
}
