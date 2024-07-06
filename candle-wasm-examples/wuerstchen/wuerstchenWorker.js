import init, {Model} from "./build/m.js";

Error.stackTraceLimit = 50;

class Wuertchen {
  static instance = {};

  static async getInstance(weightsURL, modelID, tokenizerURL, useWgpu) {
    // load individual modelID only once
    if (!this.instance[modelID]) {
      await init();
      self.postMessage({ status: "loading", message: "Loading Model" });

      this.instance[modelID] = new Model();
    }
    return this.instance[modelID];
  }
}

let controller = null;
self.addEventListener("message", (event) => {
  console.log("got Message");
  console.log(event);
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
    prompt,
    temp,
    top_p,
    repeatPenalty,
    seed,
    maxSeqLen,
    useWgpu
  } = data;
  try {
    self.postMessage({ status: "loading", message: "Starting wuerstchen" });

    console.log("generate useWgpu:")
    console.log(useWgpu)
    console.log((useWgpu === 'true'))
    console.log((useWgpu === 'True'))

    self.postMessage({ status: "loading", message: "Initializing model" });
    // const firstToken = await model.init_with_prompt(
    //   prompt,
    //   temp,
    //   top_p,
    //   repeatPenalty,
    //   seed
    // );

    //const seq_len = model.get_seq_len();

    // let sentence = firstToken;
    // //let maxTokens = maxSeqLen ? maxSeqLen : seq_len - prompt.length - 1;
    // let startTime = performance.now();
    // let tokensCount = 0;
    // while (tokensCount < maxTokens) {
    //   await new Promise(async (resolve) => {
    //     if (controller && controller.signal.aborted) {
    //       self.postMessage({
    //         status: "aborted",
    //         message: "Aborted",
    //         output: prompt + sentence,
    //       });
    //       return;
    //     }
    //     const token = await model.next_token();
    //     const tokensSec =
    //       ((tokensCount + 1) / (performance.now() - startTime)) * 1000;

    //     sentence += token;
    //     self.postMessage({
    //       status: "generating",
    //       message: "Generating token",
    //       token: token,
    //       sentence: sentence,
    //       totalTime: performance.now() - startTime,
    //       tokensSec,
    //       prompt: prompt,
    //     });
    //     setTimeout(resolve, 0);
    //   });
    //   tokensCount++;
    // }
    let sentence = "Das ist ein Test";
    self.postMessage({
      status: "complete",
      message: "complete",
      output: prompt + sentence,
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
}
