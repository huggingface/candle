import init, { Model } from "./build/m.js";

async function fetchArrayBuffer(url) {
  const cacheName = "phi-mixformer-candle-cache";
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
async function concatenateArrayBuffers(urls) {
  const arrayBuffers = await Promise.all(urls.map(url => fetchArrayBuffer(url)));

  let totalLength = arrayBuffers.reduce((acc, arrayBuffer) => acc + arrayBuffer.byteLength, 0);
  let concatenatedBuffer = new Uint8Array(totalLength);

  let offset = 0;
  arrayBuffers.forEach(buffer => {
    concatenatedBuffer.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  });
  return concatenatedBuffer;
}

class Phi {
  static instance = {};

  static async getInstance(
    weightsURL,
    modelID,
    tokenizerURL,
    configURL,
    quantized
  ) {
    // load individual modelID only once
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({ status: "loading", message: "Loading Model" });
      const [weightsArrayU8, tokenizerArrayU8, configArrayU8] =
        await Promise.all([
          weightsURL instanceof Array ? concatenateArrayBuffers(weightsURL) : fetchArrayBuffer(weightsURL),
          fetchArrayBuffer(tokenizerURL),
          fetchArrayBuffer(configURL),
        ]);

      this.instance[modelID] = new Model(
        weightsArrayU8,
        tokenizerArrayU8,
        configArrayU8,
        quantized
      );
    }
    return this.instance[modelID];
  }
}

let controller = null;
self.addEventListener("message", (event) => {
  if (event.data.command === "start") {
    controller = new AbortController();
    generate(event.data);
  } else if (event.data.command === "abort") {
    controller.abort();
  }
});

async function generate(data) {
  const {
    weightsURL,
    modelID,
    tokenizerURL,
    configURL,
    quantized,
    prompt,
    temp,
    top_p,
    repeatPenalty,
    seed,
    maxSeqLen,
  } = data;
  try {
    self.postMessage({ status: "loading", message: "Starting Phi" });
    const model = await Phi.getInstance(
      weightsURL,
      modelID,
      tokenizerURL,
      configURL,
      quantized
    );

    self.postMessage({ status: "loading", message: "Initializing model" });
    const firstToken = model.init_with_prompt(
      prompt,
      temp,
      top_p,
      repeatPenalty,
      64,
      BigInt(seed)
    );
    const seq_len = 2048;

    let sentence = firstToken;
    let maxTokens = maxSeqLen ? maxSeqLen : seq_len - prompt.length - 1;
    let startTime = performance.now();
    let tokensCount = 0;
    while (tokensCount < maxTokens) {
      await new Promise(async (resolve) => {
        if (controller && controller.signal.aborted) {
          self.postMessage({
            status: "aborted",
            message: "Aborted",
            output: prompt + sentence,
          });
          return;
        }
        const token = await model.next_token();
        if (token === "<|endoftext|>") {
          self.postMessage({
            status: "complete",
            message: "complete",
            output: prompt + sentence,
          });
          return;
        }
        const tokensSec =
          ((tokensCount + 1) / (performance.now() - startTime)) * 1000;

        sentence += token;
        self.postMessage({
          status: "generating",
          message: "Generating token",
          token: token,
          sentence: sentence,
          totalTime: performance.now() - startTime,
          tokensSec,
          prompt: prompt,
        });
        setTimeout(resolve, 0);
      });
      tokensCount++;
    }
    self.postMessage({
      status: "complete",
      message: "complete",
      output: prompt + sentence,
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
}
