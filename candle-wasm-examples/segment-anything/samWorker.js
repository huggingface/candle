//load the candle SAM Model wasm module
import init, { Model } from "./build/m.js";

async function fetchArrayBuffer(url, cacheModel = true) {
  if (!cacheModel)
    return new Uint8Array(await (await fetch(url)).arrayBuffer());
  const cacheName = "sam-candle-cache";
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
class SAMModel {
  static instance = {};
  // keep current image embeddings state
  static imageArrayHash = {};
  // Add a new property to hold the current modelID
  static currentModelID = null;

  static async getInstance(modelURL, modelID) {
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({
        status: "loading",
        message: `Loading Model ${modelID}`,
      });
      const weightsArrayU8 = await fetchArrayBuffer(modelURL);
      this.instance[modelID] = new Model(
        weightsArrayU8,
        /tiny|mobile/.test(modelID)
      );
    } else {
      self.postMessage({ status: "loading", message: "Model Already Loaded" });
    }
    // Set the current modelID to the modelID that was passed in
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

async function createImageCanvas(
  { mask_shape, mask_data }, // mask
  { original_width, original_height, width, height } // original image
) {
  const [_, __, shape_width, shape_height] = mask_shape;
  const maskCanvas = new OffscreenCanvas(shape_width, shape_height); // canvas for mask
  const maskCtx = maskCanvas.getContext("2d");
  const canvas = new OffscreenCanvas(original_width, original_height); // canvas for creating mask with original image size
  const ctx = canvas.getContext("2d");

  const imageData = maskCtx.createImageData(
    maskCanvas.width,
    maskCanvas.height
  );
  const data = imageData.data;

  for (let p = 0; p < data.length; p += 4) {
    data[p] = 0;
    data[p + 1] = 0;
    data[p + 2] = 0;
    data[p + 3] = mask_data[p / 4] * 255;
  }
  maskCtx.putImageData(imageData, 0, 0);

  let sx, sy;
  if (original_height < original_width) {
    sy = original_height / original_width;
    sx = 1;
  } else {
    sy = 1;
    sx = original_width / original_height;
  }
  ctx.drawImage(
    maskCanvas,
    0,
    0,
    maskCanvas.width * sx,
    maskCanvas.height * sy,
    0,
    0,
    original_width,
    original_height
  );

  const blob = await canvas.convertToBlob();
  return URL.createObjectURL(blob);
}

self.addEventListener("message", async (event) => {
  const { modelURL, modelID, imageURL, points } = event.data;
  try {
    self.postMessage({ status: "loading", message: "Starting SAM" });
    const sam = await SAMModel.getInstance(modelURL, modelID);

    self.postMessage({ status: "loading", message: "Loading Image" });
    const imageArrayU8 = await fetchArrayBuffer(imageURL, false);

    self.postMessage({ status: "embedding", message: "Creating Embeddings" });
    SAMModel.setImageEmbeddings(imageArrayU8);
    if (!points) {
      // no points only do the embeddings
      self.postMessage({
        status: "complete-embedding",
        message: "Embeddings Complete",
      });
      return;
    }

    self.postMessage({ status: "segmenting", message: "Segmenting" });
    const { mask, image } = sam.mask_for_point({ points });
    const maskDataURL = await createImageCanvas(mask, image);
    // Send the segment back to the main thread as JSON
    self.postMessage({
      status: "complete",
      message: "Segmentation Complete",
      output: { maskURL: maskDataURL },
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
});
