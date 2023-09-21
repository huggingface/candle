export async function encodeSentences(
  worker,
  weightsURL,
  tokenizerURL,
  configURL,
  modelID,
  sentences,
  updateStatus,
  normalize_embeddings
) {
  return new Promise((resolve, reject) => {
    worker.postMessage({
      weightsURL,
      tokenizerURL,
      configURL,
      modelID,
      sentences,
      normalize_embeddings,
    });
    function messageHandler(event) {
      if ("error" in event.data) {
        worker.removeEventListener("message", messageHandler);
        reject(new Error(event.data.error));
      }
      if (event.data.status === "complete") {
        worker.removeEventListener("message", messageHandler);
        resolve(event.data);
      }
      if (updateStatus) updateStatus(event.data);
    }
    worker.addEventListener("message", messageHandler);
  });
}

export async function decodeSentence(
  worker,
  weightsURL,
  tokenizerURL,
  configURL,
  modelID,
  prompt,
  params,
  updateStatus
) {
  return new Promise((resolve, reject) => {
    worker.postMessage({
      weightsURL,
      tokenizerURL,
      configURL,
      modelID,
      prompt,
      params,
    });
    function messageHandler(event) {
      if ("error" in event.data) {
        worker.removeEventListener("message", messageHandler);
        reject(new Error(event.data.error));
      }
      if (event.data.status === "complete") {
        worker.removeEventListener("message", messageHandler);
        resolve(event.data);
      }
      if (updateStatus) updateStatus(event.data);
    }
    worker.addEventListener("message", messageHandler);
  });
}

const MODELS = {
  t5_small: {
    base_url: "https://huggingface.co/t5-small/resolve/main/",
  },
  t5_base: {
    base_url: "https://huggingface.co/t5-base/resolve/main/",
  },
  flan_t5_small: {
    base_url:
      "https://huggingface.co/google/flan-t5-small/resolve/refs%2Fpr%2F14/",
  },
  flan_t5_base: {
    base_url: "https://huggingface.co/google/flan-t5-base/resolve/main/",
  },
};
export function getModelInfo(id) {
  return {
    modelURL: MODELS[id].base_url + "model.safetensors",
    configURL: MODELS[id].base_url + "config.json",
    tokenizerURL: MODELS[id].base_url + "tokenizer.json",
  };
}
