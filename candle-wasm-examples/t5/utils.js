export async function decodePrompt(
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

export async function getEmbeddings(
  worker,
  weightsURL,
  tokenizerURL,
  configURL,
  modelID,
  sentences,
  updateStatus = null
) {
  return new Promise((resolve, reject) => {
    worker.postMessage({
      weightsURL,
      tokenizerURL,
      configURL,
      modelID,
      sentences,
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
    base_url: "https://huggingface.co/t5-small/resolve/refs%2Fpr%2F15/",
  },
};
export function getModelInfo(id) {
  return {
    modelURL: MODELS[id].base_url + "model.safetensors",
    configURL: MODELS[id].base_url + "config.json",
    tokenizerURL: MODELS[id].base_url + "tokenizer.json",
  };
}

export function cosineSimilarity(vec1, vec2) {
  const dot = vec1.reduce((acc, val, i) => acc + val * vec2[i], 0);
  const a = Math.sqrt(vec1.reduce((acc, val) => acc + val * val, 0));
  const b = Math.sqrt(vec2.reduce((acc, val) => acc + val * val, 0));
  return dot / (a * b);
}
export async function getWikiText(article) {
  // thanks to wikipedia for the API
  const URL = `https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exlimit=1&titles=${article}&explaintext=1&exsectionformat=plain&format=json&origin=*`;
  return fetch(URL, {
    method: "GET",
    headers: {
      Accept: "application/json",
    },
  })
    .then((r) => r.json())
    .then((data) => {
      const pages = data.query.pages;
      const pageId = Object.keys(pages)[0];
      const extract = pages[pageId].extract;
      if (extract === undefined || extract === "") {
        throw new Error("No article found");
      }
      return extract;
    })
    .catch((error) => console.error("Error:", error));
}
