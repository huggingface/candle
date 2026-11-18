/** @type {boolean}*/
let loaded = false;
let isLoading = false;
const worker = new Worker("./worker.js", { type: "module" });
const modelName = "MoritzLaurer_ModernBERT_base_zeroshot_v2_0";

/**
 * @param {string} id HTMLElement's id
 */
function elementNotFound(id) {
    return new Error(`Element with id=${id} not found.`);
}

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("form");
    const statusDiv = document.getElementById("status");
    const progressContainer = document.getElementById("progressContainer");
    const progressBar = document.getElementById("progressBar");
    const spinner = document.getElementById("spinner");

    const selectExample = document.getElementById("select-example");

    if (
        !form ||
        !statusDiv ||
        !progressContainer ||
        !progressBar ||
        !spinner ||
        !selectExample
    ) {
        console.error("Required elements not found");
        return;
    }

    statusDiv.textContent = "Ready to load model";
    selectExample.addEventListener("change", (event) => {
        const textInput = /** @type {HTMLInputElement | null} */ (
            document.getElementById("text")
        );
        const labelsInput = /** @type {HTMLInputElement | null} */ (
            document.getElementById("labels")
        );
        if (!textInput || !labelsInput) {
            console.error("Form inputs not found");
            return;
        }

        const value = /** @type {HTMLSelectElement} */ (event.target).value;
        let example = "";
        let hypothesis = "";
        switch (value) {
            case "1":
                example = "Financial markets gained 5 %.";
                hypothesis = "This sentence is positive.";
                break;
            case "2":
                example = "Financial markets went down by 5%.";
                hypothesis = "This sentence is negative.";
                break;
            case "":
                return;
            default:
                console.warn("Unexpected select value:", value);
                return;
        }
        textInput.value = example;
        labelsInput.value = hypothesis;
    });

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        const textInput = /** @type {HTMLInputElement | null} */ (
            document.getElementById("text")
        );
        const labelsInput = /** @type {HTMLInputElement | null} */ (
            document.getElementById("labels")
        );
        const multiLabelInput = /** @type {HTMLInputElement | null} */ (
            document.getElementById("multilabel")
        );

        if (!textInput || !labelsInput || !multiLabelInput) {
            console.error("Form inputs not found");
            return;
        }

        const text = textInput.value.trim();
        if (!text) {
            statusDiv.textContent = "Please enter some text";
            return;
        }

        const labels = labelsInput.value.trim()
            ? labelsInput.value.split(",").map((label) => label.trim())
            : [];

        if (labels.length === 0) {
            statusDiv.textContent = "Please enter at least one label";
            return;
        }

        const multiLabel = multiLabelInput.checked;

        if (!loaded && !isLoading) {
            isLoading = true;
            statusDiv.textContent = "Loading model...";
            spinner.classList.remove("hidden");
            progressContainer.classList.remove("hidden");

            worker.postMessage({
                type: "load",
                data: {
                    modelName: modelName,
                },
            });
            return;
        }

        if (isLoading) {
            statusDiv.textContent = "Please wait while the model loads...";
            return;
        }

        spinner.classList.remove("hidden");
        statusDiv.textContent = "Processing...";
        console.log("Labels sent");
        console.log(labels);
        worker.postMessage({
            type: "infer",
            data: {
                text: [text],
                hypothesis: labels,
                multiLabel: multiLabel,
            },
        });
    });
});

worker.onmessage = (event) => {
    const { type, message } = event.data;
    const progressContainer = document.getElementById("progressContainer");
    const statusDiv = document.getElementById("status");
    const progressBar = document.getElementById("progressBar");
    const spinner = document.getElementById("spinner");

    if (!progressContainer || !statusDiv || !progressBar || !spinner) {
        throw elementNotFound("Required elements");
    }

    switch (type) {
        case "initialized":
            statusDiv.textContent = "Ready to start";
            break;

        case "loading":
            progressContainer.classList.remove("hidden");
            spinner.classList.remove("hidden");
            break;

        case "download":
            const { file, bytes, total } = message;
            statusDiv.textContent = `Downloading: ${file}`;
            const progress = (bytes / total) * 100;
            progressBar.style.width = `${progress}%`;
            break;

        case "ready":
            progressContainer.classList.add("hidden");
            loaded = true;
            isLoading = false;
            statusDiv.textContent = "Model loaded - Ready to process";
            spinner.classList.add("hidden");
            progressBar.style.width = "0%";
            /** @type {HTMLFormElement}*/ (
                document.getElementById("form")
            )?.requestSubmit();
            break;

        case "result":
            spinner.classList.add("hidden");
            if (message.error) {
                statusDiv.textContent = `Error: ${message.error}`;
            } else {
                const { probs, labels } = message;
                const results = labels.map(
                    (label, i) =>
                        `${label}: ${(probs[0][i] * 100).toFixed(2)}%`,
                );
                statusDiv.innerHTML = "Results:<br>" + results.join("<br>");
            }
            break;

        case "error":
            isLoading = false;
            spinner.classList.add("hidden");
            statusDiv.textContent = `Error: ${message || "Unknown error occurred"}`;
            break;

        case "disposed":
            loaded = false;
            isLoading = false;
            statusDiv.textContent = "Model disposed";
            break;
    }
};
