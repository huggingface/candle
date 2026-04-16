//load the candle Whisper decoder wasm module
import init, { Decoder } from "./build/m.js";

async function fetchArrayBuffer(url) {
  const cacheName = "whisper-candle-cache";
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
class Whisper {
  static instance = {};
  // Retrieve the Whisper model. When called for the first time,
  // this will load the model and save it for future use.
  static async getInstance(params) {
    const {
      weightsURL,
      modelID,
      tokenizerURL,
      mel_filtersURL,
      configURL,
      quantized,
      is_multilingual,
      timestamps,
      task,
      language,
    } = params;
    // load individual modelID only once
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({ status: "loading", message: "Loading Model" });
      const [
        weightsArrayU8,
        tokenizerArrayU8,
        mel_filtersArrayU8,
        configArrayU8,
      ] = await Promise.all([
        fetchArrayBuffer(weightsURL),
        fetchArrayBuffer(tokenizerURL),
        fetchArrayBuffer(mel_filtersURL),
        fetchArrayBuffer(configURL),
      ]);

      this.instance[modelID] = new Decoder(
        weightsArrayU8,
        tokenizerArrayU8,
        mel_filtersArrayU8,
        configArrayU8,
        quantized,
        is_multilingual,
        timestamps,
        task,
        language
      );
    } else {
      self.postMessage({ status: "loading", message: "Model Already Loaded" });
    }
    return this.instance[modelID];
  }
}

self.addEventListener("message", async (event) => {
  const {
    weightsURL,
    modelID,
    tokenizerURL,
    configURL,
    mel_filtersURL,
    audioURL,
  } = event.data;
  try {
    self.postMessage({ status: "decoding", message: "Starting Decoder" });
    let quantized = false;
    if (modelID.includes("quantized")) {
      quantized = true;
    }
    let is_multilingual = false;
    if (modelID.includes("multilingual")) {
      is_multilingual = true;
    }
    let timestamps = true;
    const decoder = await Whisper.getInstance({
      weightsURL,
      modelID,
      tokenizerURL,
      mel_filtersURL,
      configURL,
      quantized,
      is_multilingual,
      timestamps,
      task: null,
      language: null,
    });

    self.postMessage({ status: "decoding", message: "Loading Audio" });
    const audioArrayU8 = await fetchArrayBuffer(audioURL);

    self.postMessage({ status: "decoding", message: "Running Decoder..." });
    const segments = decoder.decode(audioArrayU8);

    // Send the segment back to the main thread as JSON
    self.postMessage({
      status: "complete",
      message: "complete",
      output: JSON.parse(segments),
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
});
