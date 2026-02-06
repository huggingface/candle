import init, { Gemma3Embedder } from "./pkg/candle_wasm_example_quant_embed_gemma.js";

async function fetchArrayU8(url) {
  const cacheName = "embeddinggemma-candle-cache";
  const cache = await caches.open(cacheName);

  const cached = await cache.match(url);
  if (cached) {
    const data = await cached.arrayBuffer();
    return new Uint8Array(data);
  }

  const res = await fetch(url, { cache: "force-cache" });
  cache.put(url, res.clone());
  return new Uint8Array(await res.arrayBuffer());
}

class Gemma {
  static instance = {};

  static async getInstance(modelURL, dense1URL, dense2URL, tokenizerURL, configURL, modelID) {
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({ status: "loading", message: "Loading model assets..." });

      const [modelU8, dense1U8, dense2U8, tokU8, cfgU8] = await Promise.all([
        fetchArrayU8(modelURL),
        fetchArrayU8(dense1URL),
        fetchArrayU8(dense2URL),
        fetchArrayU8(tokenizerURL),
        fetchArrayU8(configURL),
      ]);

      // Gemma3Embedder expects bytes (Uint8Array) for all inputs in your current setup
      this.instance[modelID] = new Gemma3Embedder(modelU8, dense1U8, dense2U8, tokU8, cfgU8);

      self.postMessage({ status: "ready", message: "Model loaded" });
    } else {
      self.postMessage({ status: "ready", message: "Model already loaded" });
    }
    return this.instance[modelID];
  }
}

self.addEventListener("message", async (event) => {
  const { modelURL, dense1URL, dense2URL, tokenizerURL, configURL, modelID, sentences } = event.data;

  try {
    const model = await Gemma.getInstance(
      modelURL,
      dense1URL,
      dense2URL,
      tokenizerURL,
      configURL,
      modelID
    );

    self.postMessage({ status: "embedding", message: "Calculating embeddings..." });

    // Return one embedding per sentence
    const output = sentences.map((s) => Array.from(model.embed(s)));

    self.postMessage({
      status: "complete",
      message: "complete",
      output,
    });
  } catch (e) {
    self.postMessage({ error: String(e?.message ?? e) });
  }
});
