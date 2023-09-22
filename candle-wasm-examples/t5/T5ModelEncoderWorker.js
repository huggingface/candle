//load Candle Bert Module wasm module
let init, ModelEncoder;

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
class Encoder {
  static instance = {};

  static async getInstance(weightsURL, tokenizerURL, configURL, modelID) {
    if (modelID.includes("quantized")) {
      ({ default: init, ModelEncoder } = await import(
        "./build/m-quantized.js"
      ));
    } else {
      ({ default: init, ModelEncoder } = await import("./build/m.js"));
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

      this.instance[modelID] = new ModelEncoder(
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
  const {
    weightsURL,
    tokenizerURL,
    configURL,
    modelID,
    sentences,
    normalize_embeddings,
  } = event.data;
  try {
    self.postMessage({ status: "ready", message: "Starting T5 Encoder" });
    const model = await Encoder.getInstance(
      weightsURL,
      tokenizerURL,
      configURL,
      modelID
    );
    self.postMessage({
      status: "encoding",
      message: "Encoding Sentences",
    });
    const output = model.decode({
      sentences: sentences,
      normalize_embeddings: normalize_embeddings || true,
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
