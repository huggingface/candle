import init from "./pkg/candle_wasm.js";

const runWasm = async () => {
  const candleWasm = await init("./pkg/candle_wasm_bg.wasm");
  candleWasm.test_fn();
};
runWasm();
