import init, { Model } from "./build/m.js";

async function fetchArrayBuffer(url) {
  const res = await fetch(url, {
    cache: "force-cache",
  });
  const data = await res.arrayBuffer();
  return new Uint8Array(data);
}

class Llama2C {
  static instance = {};

  static async getInstance(weightsURL, modelID, tokenizerURL) {
    // load individual modelID only once
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({ status: "loading", message: "Loading Model" });

      const [weightsArrayU8, tokenizerArrayU8] = await Promise.all([
        fetchArrayBuffer(weightsURL),
        fetchArrayBuffer(tokenizerURL),
      ]);

      this.instance[modelID] = new Model(weightsArrayU8, tokenizerArrayU8);
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
    prompt,
    temp,
    repeatPenalty,
    seed,
  } = event.data;
  try {
    self.postMessage({ status: "loading", message: "Starting llama2.c" });
    const model = await Llama2C.getInstance(weightsURL, modelID, tokenizerURL);

    self.postMessage({ status: "loading", message: "Inititng llama model" });
    model.init_with_prompt(prompt, temp, repeatPenalty, seed);

    const seq_len = model.get_seq_len();
    console.log("seq_len", seq_len);
    let setence = "";

    let max_tokens = seq_len - prompt.length - 1;
    while (max_tokens--) {
      const token = await model.next_token();

      setence += token;
      self.postMessage({
        status: "generating",
        message: "Generating token",
        token: token,
        setence: setence,
        prompt: prompt,
      });
    }
    self.postMessage({
      status: "complete",
      message: "complete",
      output: prompt + setence,
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
});
