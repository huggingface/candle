import snarkdown from "https://cdn.skypack.dev/snarkdown";
import hljs from "https://cdn.skypack.dev/highlight.js";
// models base url
const MODELS = {
  moondream2_q4k: {
    base_url:
      "https://huggingface.co/santiagomed/candle-moondream/resolve/main/",
    model: "model-q4_0.gguf",
    tokenizer: "tokenizer.json",
    quantized: true,
    size: "1.51 GB",
  },
};

const moodreamWorker = new Worker("./moondreamWorker.js", {
  type: "module",
});

async function generateSequence(controller) {
  const getValue = (id) => document.querySelector(`#${id}`).value;
  const modelID = getValue("model");
  const model = MODELS[modelID];
  const weightsURL =
    model.model instanceof Array
      ? model.model.map((m) => model.base_url + m)
      : model.base_url + model.model;
  const tokenizerURL = model.base_url + model.tokenizer;

  const prompt = getValue("prompt").trim();
  const temperature = getValue("temperature");
  const topP = getValue("top-p");
  const repeatPenalty = getValue("repeat_penalty");
  const seed = getValue("seed");
  const maxSeqLen = getValue("max-seq");

  if (prompt?.value?.trim() === "") {
    return;
  }

  function updateStatus(data) {
    const outStatus = document.querySelector("#output-status");
    const outGen = document.querySelector("#output-generation");
    const outCounter = document.querySelector("#output-counter");

    switch (data.status) {
      case "loading":
        outStatus.hidden = false;
        outStatus.textContent = data.message;
        outGen.hidden = true;
        outCounter.hidden = true;
        break;
      case "generating":
        const { message, prompt, sentence, tokensSec, totalTime } = data;
        outStatus.hidden = true;
        outCounter.hidden = false;
        outGen.hidden = false;
        outGen.innerHTML = snarkdown(prompt + sentence);
        outCounter.innerHTML = `${(totalTime / 1000).toFixed(
          2
        )}s (${tokensSec.toFixed(2)} tok/s)`;
        hljs.highlightAll();
        break;
      case "complete":
        outStatus.hidden = true;
        outGen.hidden = false;
        break;
    }
  }

  return new Promise((resolve, reject) => {
    moodreamWorker.postMessage({
      weightsURL,
      modelID,
      tokenizerURL,
      quantized: model.quantized,
      imageURL: currentImageURL,
      prompt,
      temp: temperature,
      top_p: topP,
      repeatPenalty,
      seed: seed,
      maxSeqLen,
      verbose_prompt: false,
      command: "start",
    });

    const handleAbort = () => {
      moodreamWorker.postMessage({ command: "abort" });
    };
    const handleMessage = (event) => {
      const { status, error, message, prompt, sentence } = event.data;
      if (status) updateStatus(event.data);
      if (error) {
        moodreamWorker.removeEventListener("message", handleMessage);
        reject(new Error(error));
      }
      if (status === "aborted") {
        moodreamWorker.removeEventListener("message", handleMessage);
        resolve(event.data);
      }
      if (status === "complete") {
        moodreamWorker.removeEventListener("message", handleMessage);
        resolve(event.data);
      }
    };

    controller.signal.addEventListener("abort", handleAbort);
    moodreamWorker.addEventListener("message", handleMessage);
  });
}

const form = document.querySelector("#form");
const prompt = document.querySelector("#prompt");
const runBtn = document.querySelector("#run");
const modelSelect = document.querySelector("#model");
const dropArea = document.querySelector("#drop-area");
const canvas = document.querySelector("#canvas");
const ctxCanvas = canvas.getContext("2d");
const fileUpload = document.querySelector("#file-upload");
const clearImgBtn = document.querySelector("#clear-img-btn");
const imagesExamples = document.querySelector("#image-select");

let currentImageURL = null;
let runController = new AbortController();
let isRunning = false;

document.addEventListener("DOMContentLoaded", () => {
  for (const [id, model] of Object.entries(MODELS)) {
    const option = document.createElement("option");
    option.value = id;
    option.innerText = `${id} (${model.size})`;
    modelSelect.appendChild(option);
  }
  const query = new URLSearchParams(window.location.search);
  const modelID = query.get("model");
  if (modelID) {
    modelSelect.value = modelID;
  } else {
    modelSelect.value = "moondream2_q4k";
  }
});

imagesExamples.addEventListener("click", (e) => {
  // if (isEmbedding || isSegmenting) {
  //   return;
  // }
  const target = e.target;
  if (target.nodeName === "IMG") {
    const href = target.src;
    clearImageCanvas();
    currentImageURL = href;
    drawImageCanvas(href);
  }
});
modelSelect.addEventListener("change", (e) => {
  const query = new URLSearchParams(window.location.search);
  query.set("model", e.target.value);
  window.history.replaceState({}, "", `${window.location.pathname}?${query}`);
  window.parent.postMessage({ queryString: "?" + query }, "*");
  const model = MODELS[e.target.value];
  document.querySelector("#max-seq").max = model.seq_len;
  document.querySelector("#max-seq").nextElementSibling.value = 200;
});

clearImgBtn.addEventListener("click", () => {
  clearImageCanvas();
});

//add event listener to file input
fileUpload.addEventListener("input", async (e) => {
  const target = e.target;
  if (target.files.length > 0 && !target.files[0].type.includes("svg")) {
    const href = URL.createObjectURL(target.files[0]);
    clearImageCanvas();
    await drawImageCanvas(href);
  }
});
// add event listener to drop-area
dropArea.addEventListener("dragenter", (e) => {
  e.preventDefault();
  dropArea.classList.add("border-blue-700");
});
dropArea.addEventListener("dragleave", (e) => {
  e.preventDefault();
  dropArea.classList.remove("border-blue-700");
});
dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
});
dropArea.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropArea.classList.remove("border-blue-700");
  const url = e.dataTransfer.getData("text/uri-list");
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    const href = URL.createObjectURL(files[0]);
    clearImageCanvas();
    await drawImageCanvas(href);
  } else if (url) {
    clearImageCanvas();
    await drawImageCanvas(url);
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (isRunning) {
    stopRunning();
  } else {
    startRunning();
    await generateSequence(runController);
    stopRunning();
  }
});

async function drawImageCanvas(imgURL) {
  if (!imgURL) {
    throw new Error("No image URL provided");
  }
  return new Promise((resolve, reject) => {
    ctxCanvas.clearRect(0, 0, canvas.width, canvas.height);
    ctxCanvas.clearRect(0, 0, canvas.width, canvas.height);
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctxCanvas.drawImage(img, 0, 0);
      clearImgBtn.disabled = false;
      resolve(img);
    };
    img.src = imgURL;
    currentImageURL = imgURL;
  });
}

function clearImageCanvas() {
  ctxCanvas.clearRect(0, 0, canvas.width, canvas.height);
  clearImgBtn.disabled = true;
  canvas.parentElement.style.height = "auto";
  currentImageURL = null;
  canvas.width = 0;
  canvas.height = 0;
}

function startRunning() {
  isRunning = true;
  runBtn.textContent = "Stop";
  prompt.disabled = true;
}

function stopRunning() {
  runController.abort();
  runController = new AbortController();
  runBtn.textContent = "Run";
  isRunning = false;
  prompt.disabled = false;
}

prompt.addEventListener("input", (e) => {
  runBtn.disabled = false;
});
