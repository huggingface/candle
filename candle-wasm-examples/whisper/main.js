import init, { run_app } from './pkg/candle_wasm_example_whisper.js';
async function main() {
   await init('/pkg/candle_wasm_example_whisper_bg.wasm');
   run_app();
}
main()
