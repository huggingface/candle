export async function extractEmbeddings(
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

export async function generateText(
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
export const MODELS = {
  t5_small: {
    size: "242 MB",
    base_url: "https://huggingface.co/t5-small/resolve/main/",
    tasks: {
      translation_en_to_de: {
        prefix: "translate English to German: ",
        max_length: 300,
      },
      translation_en_to_fr: {
        prefix: "translate English to French: ",
        max_length: 300,
      },
      translation_en_to_ro: {
        prefix: "translate English to Romanian: ",
        max_length: 300,
      },
      summarization: { prefix: "summarize: ", max_length: 200 },
    },
  },
  flan_t5_small: {
    size: "308 MB",
    base_url:
      "https://huggingface.co/google/flan-t5-small/resolve/refs%2Fpr%2F14/",
    tasks: {
      translation_en_to_de: {
        prefix: "translate English to German: ",
        max_length: 300,
      },
      translation_en_to_fr: {
        prefix: "translate English to French: ",
        max_length: 300,
      },
      translation_en_to_ro: {
        prefix: "translate English to Romanian: ",
        max_length: 300,
      },
      summarization: { prefix: "summarize: ", max_length: 200 },
    },
  },
};
export function getModelInfo(id, taskID) {
  return {
    modelURL: MODELS[id].base_url + "model.safetensors",
    configURL: MODELS[id].base_url + "config.json",
    tokenizerURL: MODELS[id].base_url + "tokenizer.json",
    maxLength: MODELS[id].tasks[taskID].max_length,
  };
}
