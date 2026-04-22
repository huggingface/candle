import init, { Model } from "./build/m.js";

async function fetchArrayBuffer(url, cacheModel = true) {
  if (!cacheModel)
    return new Uint8Array(await (await fetch(url)).arrayBuffer());
  const cacheName = "moondream-candle-cache";
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
  const arrayBuffers = await Promise.all(
    urls.map((url) => fetchArrayBuffer(url))
  );

  let totalLength = arrayBuffers.reduce(
    (acc, arrayBuffer) => acc + arrayBuffer.byteLength,
    0
  );
  let concatenatedBuffer = new Uint8Array(totalLength);

  let offset = 0;
  arrayBuffers.forEach((buffer) => {
    concatenatedBuffer.set(new Uint8Array(buffer), offset);
    offset += buffer.byteLength;
  });
  return concatenatedBuffer;
}

class Moondream {
  static imageArrayHash = {};
  static instance = {};
  static currentModelID = null;

  static async getInstance(weightsURL, modelID, tokenizerURL, quantized) {
    // load individual modelID only once
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({ status: "loading", message: "Loading Model" });
      const [weightsArrayU8, tokenizerArrayU8] = await Promise.all([
        weightsURL instanceof Array
          ? concatenateArrayBuffers(weightsURL)
          : fetchArrayBuffer(weightsURL),
        fetchArrayBuffer(tokenizerURL),
      ]);

      this.instance[modelID] = new Model(
        weightsArrayU8,
        tokenizerArrayU8,
        quantized
      );
    }
    this.currentModelID = modelID;
    return this.instance[modelID];
  }

  // Remove the modelID parameter from setImageEmbeddings
  static setImageEmbeddings(imageArrayU8) {
    // check if image embeddings are already set for this image and model
    const imageArrayHash = this.getSimpleHash(imageArrayU8);
    if (
      this.imageArrayHash[this.currentModelID] === imageArrayHash &&
      this.instance[this.currentModelID]
    ) {
      self.postMessage({
        status: "embedding",
        message: "Embeddings Already Set",
      });
      return;
    }
    this.imageArrayHash[this.currentModelID] = imageArrayHash;
    this.instance[this.currentModelID].set_image_embeddings(imageArrayU8);
    self.postMessage({ status: "embedding", message: "Embeddings Set" });
  }

  static getSimpleHash(imageArrayU8) {
    // get simple hash of imageArrayU8
    let imageArrayHash = 0;
    for (let i = 0; i < imageArrayU8.length; i += 100) {
      imageArrayHash ^= imageArrayU8[i];
    }
    return imageArrayHash.toString(16);
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
    quantized,
    imageURL,
    prompt,
    seed,
    temp,
    top_p,
    repeatPenalty,
    maxSeqLen,
    verbose_prompt,
  } = data;
  try {
    self.postMessage({ status: "loading", message: "Starting Moondream" });
    const model = await Moondream.getInstance(
      weightsURL,
      modelID,
      tokenizerURL,
      quantized
    );

    self.postMessage({ status: "loading", message: "Initializing model" });

    self.postMessage({ status: "loading", message: "Loading Image" });
    const imageArrayU8 = await fetchArrayBuffer(imageURL, false);

    self.postMessage({ status: "embedding", message: "Creating Embeddings" });
    Moondream.setImageEmbeddings(imageArrayU8);
    self.postMessage({
      status: "complete-embedding",
      message: "Embeddings Complete",
    });
    const { token, token_id } = model.init_with_image_prompt({
      prompt,
      seed: BigInt(seed),
      temp: parseFloat(temp),
      top_p: parseFloat(top_p),
      repeat_penalty: parseFloat(repeatPenalty),
      repeat_last_n: 64,
      verbose_prompt,
    });

    const seq_len = 2048;

    let sentence = token;
    let maxTokens = maxSeqLen ? maxSeqLen : seq_len - prompt.length - 1;
    let startTime = performance.now();
    let tokensCount = 0;
    while (tokensCount < maxTokens) {
      await new Promise(async (resolve) => {
        if (controller && controller.signal.aborted) {
          console.log("Aborted");
          self.postMessage({
            status: "aborted",
            message: "Aborted",
            output: prompt + sentence,
          });
          return;
        }
        const { token, token_id } = await model.next_token();
        if (token_id === 50256) {
          // <|endoftext|>
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
