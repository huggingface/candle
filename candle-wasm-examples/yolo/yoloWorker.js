//load the candle yolo wasm module
import init, { Model, ModelPose } from "./build/m.js";

async function fetchArrayBuffer(url) {
  const cacheName = "yolo-candle-cache";
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

class Yolo {
  static instance = {};
  // Retrieve the YOLO model. When called for the first time,
  // this will load the model and save it for future use.
  static async getInstance(modelID, modelURL, modelSize) {
    // load individual modelID only once
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({ status: `loading model ${modelID}:${modelSize}` });
      const weightsArrayU8 = await fetchArrayBuffer(modelURL);
      if (/pose/.test(modelID)) {
        // if pose model, use ModelPose
        this.instance[modelID] = new ModelPose(weightsArrayU8, modelSize);
      } else {
        this.instance[modelID] = new Model(weightsArrayU8, modelSize);
      }
    } else {
      self.postMessage({ status: "model already loaded" });
    }
    return this.instance[modelID];
  }
}

self.addEventListener("message", async (event) => {
  const { imageURL, modelID, modelURL, modelSize, confidence, iou_threshold } =
    event.data;
  try {
    self.postMessage({ status: "detecting" });

    const yolo = await Yolo.getInstance(modelID, modelURL, modelSize);

    self.postMessage({ status: "loading image" });
    const imgRes = await fetch(imageURL);
    const imgData = await imgRes.arrayBuffer();
    const imageArrayU8 = new Uint8Array(imgData);

    self.postMessage({ status: `running inference ${modelID}:${modelSize}` });
    const bboxes = yolo.run(imageArrayU8, confidence, iou_threshold);

    // Send the output back to the main thread as JSON
    self.postMessage({
      status: "complete",
      output: JSON.parse(bboxes),
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
});
