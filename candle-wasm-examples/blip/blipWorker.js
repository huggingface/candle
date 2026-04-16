import init, { Model } from "./build/m.js";

async function fetchArrayBuffer(url, cacheFile = true) {
  if (!cacheFile) return new Uint8Array(await (await fetch(url)).arrayBuffer());
  const cacheName = "blip-candle-cache";
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
class Blip {
  static instance = {};

  static async getInstance(
    weightsURL,
    tokenizerURL,
    configURL,
    modelID,
    quantized
  ) {
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({ status: "loading", message: "Loading Model" });
      const [weightsArrayU8, tokenizerArrayU8, configArrayU8] =
        await Promise.all([
          fetchArrayBuffer(weightsURL),
          fetchArrayBuffer(tokenizerURL),
          fetchArrayBuffer(configURL),
        ]);

      this.instance[modelID] = new Model(
        weightsArrayU8,
        tokenizerArrayU8,
        configArrayU8,
        quantized
      );
    } else {
      self.postMessage({ status: "ready", message: "Model Already Loaded" });
    }
    return this.instance[modelID];
  }
}

self.addEventListener("message", async (event) => {
  const { weightsURL, tokenizerURL, configURL, modelID, imageURL, quantized } =
    event.data;
  try {
    self.postMessage({ status: "status", message: "Loading Blip Model..." });
    const model = await Blip.getInstance(
      weightsURL,
      tokenizerURL,
      configURL,
      modelID,
      quantized
    );
    self.postMessage({
      status: "status",
      message: "Running Blip Inference...",
    });
    const imageArrayU8 = await fetchArrayBuffer(imageURL, false);
    const output = model.generate_caption_from_image(imageArrayU8);

    self.postMessage({
      status: "complete",
      message: "complete",
      output: output,
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
});
