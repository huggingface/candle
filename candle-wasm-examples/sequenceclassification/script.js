const worker = new Worker("./worker.js", { type: "module" });

/**
 * @type {Set<string>}
 */
let loaded = new Set();

/**
 * @param {string} id HTMLElement's id
 */
function elementNotFound(id) {
    return new Error(`Element with id=${id} not found.`);
}

const modelSelectEl = document.querySelector("#model");
const searchExampleEl = document.querySelector("#search-example");
const formExampleEl = document.querySelector("#form-example");
const textInput = document.getElementById("textInput");

async function getWikiText(article) {
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

formExampleEl?.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (searchExampleEl === null || textInput === null) {
        return;
    }

    // @ts-ignore
    const value = e.submitter?.value || null;

    let text = null;
    if (value === "Moon_Landing") {
        try {
            text = await getWikiText("Moon_landing");
        } catch {
            // @ts-ignore
            searchExampleEl.setCustomValidity("Invalid Wikipedia article name");
            // @ts-ignore
            searchExampleEl.reportValidity();
            return;
        }
    } else if (value === "financial") {
        text = "Financial markets went down by 5%.";
    }

    // @ts-ignore
    searchExampleEl.setCustomValidity("");
    // @ts-ignore
    textInput.value = text;
});

worker.onmessage = (event) => {
    const { type, message } = event.data;
    const statusDiv = document.getElementById("status");
    const resultPre = document.getElementById("result");
    const progressContainer = document.getElementById("progressContainer");
    const progressBar = document.getElementById("progressBar");
    const spinner = document.getElementById("spinner");

    if (statusDiv === null) {
        throw elementNotFound("status");
    }
    if (resultPre === null) {
        throw elementNotFound("result");
    }
    if (progressContainer === null) {
        throw elementNotFound("progressContainer");
    }
    if (progressBar === null) {
        throw elementNotFound("progressBar");
    }
    if (spinner === null) {
        throw elementNotFound("spinner");
    }
    if (modelSelectEl === null) {
        throw new Error("Did not find element by query selector #model");
    }

    switch (type) {
        case "initialized":
            statusDiv.textContent = "Worker initialized successfully";
            resultPre.textContent = "";
            break;
        case "loading":
            statusDiv.textContent = "Loading model...";
            resultPre.textContent = "";
            progressContainer.classList.remove("hidden");
            spinner.classList.remove("hidden");
            // @ts-ignore
            modelSelectEl.disabled = true;
            break;
        case "download":
            const { file, bytes, total } = message;
            statusDiv.textContent = `Downloading: ${file}`;
            const progress = (bytes / total) * 100;
            progressBar.style.width = `${progress}%`;
            break;
        case "ready":
            loaded.add(event.data.modelName);
            statusDiv.textContent = "Model ready";
            resultPre.textContent = "";
            progressContainer.classList.add("hidden");
            spinner.classList.add("hidden");
            progressBar.style.width = "0%";
            // @ts-ignore
            modelSelectEl.disabled = false;
            break;
        case "result":
            const { probs, labels } = message;
            if (labels) {
                const label = labels[0];
                const probabilities = probs[0];
                const idx_probability = probabilities.indexOf(
                    Math.max(...probabilities),
                );
                const probability = probabilities[idx_probability];
                resultPre.textContent = `Label: ${label}\nProbability: ${(probability * 100).toFixed(2)}%\nAmong labels: ${labels}`;
            }
            statusDiv.textContent = "Prediction complete";
            spinner.classList.add("hidden");
            break;
        case "error":
            statusDiv.textContent = `Error: ${message}`;
            resultPre.textContent = "Prediction failed";
            progressContainer.classList.add("hidden");
            spinner.classList.add("hidden");
            progressBar.style.width = "0%";
            break;
        case "disposed":
            statusDiv.textContent = "Model disposed";
            resultPre.textContent = "Waiting for prediction...";
            break;
        default:
            statusDiv.textContent = `Unknown message: ${type}`;
    }
};

modelSelectEl?.addEventListener("change", (e) => {
    const statusDiv = document.getElementById("status");
    const resultPre = document.getElementById("result");
    const progressContainer = document.getElementById("progressContainer");

    if (statusDiv === null) {
        throw elementNotFound("status");
    }
    if (resultPre === null) {
        throw elementNotFound("result");
    }
    if (progressContainer === null) {
        throw elementNotFound("progressContainer");
    }
    if (modelSelectEl === null) {
        throw new Error("Did not find element by query selector #model");
    }

    statusDiv.textContent = "Preparing to load model...";
    resultPre.textContent = "";
    progressContainer.classList.remove("hidden");

    try {
        // @ts-ignore
        const modelName = modelSelectEl.value;
        if (!modelName) {
            statusDiv.textContent = "Error: No model selected";
            return;
        }

        worker.postMessage({
            type: "dispose",
        });

        worker.postMessage({
            type: "load",
            data: {
                modelName: modelName,
            },
        });
    } catch (error) {
        statusDiv.textContent = `Error loading model: ${error.message}`;
        progressContainer.classList.add("hidden");
    }
});

document.getElementById("inputForm")?.addEventListener("submit", (e) => {
    e.preventDefault();
    const status = document.getElementById("status");
    if (status === null) {
        throw elementNotFound("status");
    }
    // @ts-ignore
    const inputValue = textInput?.value.trim();
    if (!inputValue) {
        status.textContent = "Please enter some text";
        return;
    }
    // @ts-ignore
    const selectedModel = modelSelectEl?.value;
    if (!loaded.has(selectedModel)) {
        worker.postMessage({
            type: "load",
            data: { modelName: selectedModel },
        });
    }

    worker.postMessage({
        type: "infer",
        data: { text: [inputValue] },
    });

    status.textContent = "Processing...";
    const spinner = document.getElementById("spinner");
    if (spinner === null) {
        throw elementNotFound("spinner");
    }
    spinner.classList.remove("hidden");
});

document.getElementById("clearButton")?.addEventListener("click", () => {
    const textInput = document.getElementById("textInput");
    if (textInput === null) {
        throw elementNotFound("textInput");
    }
    // @ts-ignore
    textInput.value = "";
    const result = document.getElementById("result");
    if (result === null) {
        throw elementNotFound("result");
    }
    result.textContent = "Waiting for prediction...";
    const status = document.getElementById("status");
    if (status === null) {
        throw elementNotFound("status");
    }
    status.textContent = "";
});
