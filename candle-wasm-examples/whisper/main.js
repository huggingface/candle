import init, { run_app } from './pkg/candle_wasm_example.js';
async function main() {
   await init('/pkg/candle_wasm_example_bg.wasm');
   run_app();
}
main()
