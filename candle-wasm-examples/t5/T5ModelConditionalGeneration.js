//load Candle Bert Module wasm module
let init, ModelConditionalGeneration;

async function fetchArrayBuffer(url) {
  const cacheName = "t5-candle-cache";
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(url);
  if (cachedResponse) {
    const data = await cachedResponse.arrayBuffer();
    return new Uint8Array(data);
  }
  const res = await fetch(url, { cache: "force-cache" });
  cache.put(url, res.clone());
  return new Uint8Array(await res.arrayBuffer());
}
class ConditionalGeneration {
  static instance = {};

  static async getInstance(weightsURL, tokenizerURL, configURL, modelID) {
    if (modelID.includes("quantized")) {
      ({ default: init, ModelConditionalGeneration } = await import(
        "./build/m-quantized.js"
      ));
    } else {
      ({ default: init, ModelConditionalGeneration } = await import(
        "./build/m.js"
      ));
    }
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({ status: "loading", message: "Loading Model" });
      const [weightsArrayU8, tokenizerArrayU8, configArrayU8] =
        await Promise.all([
          fetchArrayBuffer(weightsURL),
          fetchArrayBuffer(tokenizerURL),
          fetchArrayBuffer(configURL),
        ]);

      this.instance[modelID] = new ModelConditionalGeneration(
        weightsArrayU8,
        tokenizerArrayU8,
        configArrayU8
      );
    } else {
      self.postMessage({ status: "ready", message: "Model Already Loaded" });
    }
    return this.instance[modelID];
  }
}

self.addEventListener("message", async (event) => {
  const { weightsURL, tokenizerURL, configURL, modelID, prompt, params } =
    event.data;
  let {
    temperature = 0.0,
    seed = 299792458,
    repeat_penalty = 1.1,
    repeat_last_n = 64,
    top_p = 1,
  } = { ...params };
  try {
    self.postMessage({
      status: "ready",
      message: "Starting T5 Conditional Generation",
    });
    const model = await ConditionalGeneration.getInstance(
      weightsURL,
      tokenizerURL,
      configURL,
      modelID
    );
    self.postMessage({
      status: "decoding",
      message: "Decoding Prompt",
    });
    const output = model.decode({
      prompt,
      temperature,
      seed,
      top_p,
      repeat_penalty,
      repeat_last_n,
    });
    self.postMessage({
      status: "complete",
      message: "complete",
      output: output,
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
});
