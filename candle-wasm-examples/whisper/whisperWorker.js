//load the candle Whisper decoder wasm module
import init, { Decoder } from "./build/m.js";

async function fetchArrayBuffer(url) {
  const res = await fetch(url);
  const data = await res.arrayBuffer();
  return new Uint8Array(data);
}

class Whisper {
  static instance = {};
  // Retrieve the Whisper model. When called for the first time,
  // this will load the model and save it for future use.
  static async getInstance(weightsURL, modelID, tokenizerURL, mel_filtersURL) {
    // load individual modelID only once
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({ status: `loading model` });
      const [weightsArrayU8, tokenizerArrayU8, mel_filtersArrayU8] =
        await Promise.all([
          fetchArrayBuffer(weightsURL),
          fetchArrayBuffer(tokenizerURL),
          fetchArrayBuffer(mel_filtersURL),
        ]);

      this.instance[modelID] = new Decoder(
        weightsArrayU8,
        tokenizerArrayU8,
        mel_filtersArrayU8
      );
    } else {
      self.postMessage({ status: "model already loaded" });
    }
    return this.instance[modelID];
  }
}

self.addEventListener("message", async (event) => {
  const { weightsURL, modelID, tokenizerURL, mel_filtersURL, audioURL } =
    event.data;
  try {
    self.postMessage({ status: "starting decoder" });

    const decoder = await Whisper.getInstance(
      weightsURL,
      modelID,
      tokenizerURL,
      mel_filtersURL
    );

    self.postMessage({ status: "loading audio" });
    const audioArrayU8 = await fetchArrayBuffer(audioURL);

    self.postMessage({ status: `running decoder` });
    const segments = decoder.decode(audioArrayU8);

    // Send the segment back to the main thread as JSON
    self.postMessage({
      status: "complete",
      output: JSON.parse(segments),
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
});
