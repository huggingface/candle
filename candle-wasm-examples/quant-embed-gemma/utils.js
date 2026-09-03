export async function getEmbeddings(
    worker,
    modelURL,
    dense1URL,
    dense2URL,
    tokenizerURL,
    configURL,
    modelID,
    sentences,
    updateStatus = null
  ) {
    return new Promise((resolve, reject) => {
      worker.postMessage({
        modelURL,
        dense1URL,
        dense2URL,
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
  
  // These URLs are served by serve.py
  const MODELS = {
    embeddinggemma_300m_q8: {
      modelURL: "/model-q8.gguf",
    },
    embeddinggemma_300m_q4: {
      modelURL: "/model-q4.gguf",
    },
  };
  
  export function getModelInfo(id) {
    return {
      modelURL: MODELS[id].modelURL,
      dense1URL: "/dense1.safetensors",
      dense2URL: "/dense2.safetensors",
      tokenizerURL: "/tokenizer.json",
      configURL: "/config.json",
    };
  }
  
  export function cosineSimilarity(vec1, vec2) {
    let dot = 0;
    let a2 = 0;
    let b2 = 0;
    for (let i = 0; i < vec1.length; i++) {
      const a = vec1[i];
      const b = vec2[i];
      dot += a * b;
      a2 += a * a;
      b2 += b * b;
    }
    return dot / (Math.sqrt(a2) * Math.sqrt(b2));
  }
  
  export async function getWikiText(article) {
    const URL = `https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exlimit=1&titles=${encodeURIComponent(
      article
    )}&explaintext=1&exsectionformat=plain&format=json&origin=*`;
  
    const data = await fetch(URL, { headers: { Accept: "application/json" } }).then((r) => r.json());
    const pages = data.query.pages;
    const pageId = Object.keys(pages)[0];
    const extract = pages[pageId].extract;
    if (!extract) throw new Error("No article found");
    return extract;
  }
  