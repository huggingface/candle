/**
 * @typedef {Uint8Array} U8Array
 * @description An array of unsigned 8-bit integers (0-255).
 */
import init, { ModernBertPredictor } from "./build/m.js";

/**
 * Fetches an array buffer from a URL, with caching support and progress callback.
 * @param {string} url - The URL to fetch the data from.
 * @param {function} onProgress - A callback function to track progress.
 * @returns {Promise<U8Array>} - A Uint8Array containing the fetched data.
 */
async function fetchArrayBuffer(url, onProgress) {
    try {
        const res = await fetch(url, { cache: "force-cache" });
        if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);

        const contentLength = res.headers.get("Content-Length");
        if (!contentLength) {
            throw new Error("Content-Length response header missing");
        }

        const total = parseInt(contentLength, 10);
        let loaded = 0;

        const reader = res.body?.getReader();
        const chunks = [];

        while (true) {
            const { done, value } = (await reader?.read()) ?? {
                done: true,
                value: undefined,
            };
            if (done) break;

            chunks.push(value);
            loaded += value.length;
            onProgress(loaded, total);
        }

        // Concatenate chunks into a single Uint8Array
        const totalLength = chunks.reduce(
            (sum, chunk) => sum + chunk.length,
            0,
        );
        const result = new Uint8Array(totalLength);
        let position = 0;
        for (const chunk of chunks) {
            result.set(chunk, position);
            position += chunk.length;
        }

        return result;
    } catch (error) {
        self.postMessage({
            type: "error",
            message: `Fetch failed: ${error.message}`,
        });
        throw error;
    }
}

/**
 * @class Model
 * @description A class representing a model instance with configuration, model, and vocabulary.
 */
class Model {
    /** @type {ModernBertPredictor | undefined} */
    static loadedModel = undefined;
    constructor() {
        /** @type {string | undefined} */
        this.configURL = undefined;
        /** @type {string | undefined} */
        this.modelURL = undefined;
        /** @type {string | undefined} */
        this.vocabURL = undefined;
        /** @type {string | undefined} */
        this.tokenizerConfigURL = undefined;
        /** @type {string | undefined} */
        this.specialTokensMapURL = undefined;

        /** @type {U8Array | undefined} vocab - The vocabulary data for the tokenizer.*/
        this.vocab = undefined;
        /**@type {U8Array | undefined} model - The safetensor model data.*/
        this.model = undefined;
        /** @type {U8Array | undefined} config - The model configuration data.*/
        this.config = undefined;
        /** @type {U8Array | undefined} config - The model configuration data.*/
        this.tokenizerConfig = undefined;
        /** @type {U8Array | undefined} config - The model configuration data.*/
        this.specialTokensMap = undefined;
    }
    /**
       * Creates a new Model instance.
       
       * 
       *      */
    async loadFiles(modelFiles) {
        /**
         * @param {string} file
         */
        const createDownloadHandler = (file) => {
            /**
             * @param {number} bytesLoaded
             * @param {number} totalBytes
             **/
            const handler = (bytesLoaded, totalBytes) => {
                self.postMessage({
                    type: "download",
                    message: {
                        file: file,
                        bytes: bytesLoaded,
                        total: totalBytes,
                    },
                });
            };
            return handler;
        };
        const {
            configURL,
            modelURL,
            vocabURL,
            specialTokensMapURL,
            tokenizerConfigURL,
        } = modelFiles;
        this.configURL = configURL;
        this.vocabURL = vocabURL;
        this.modelURL = modelURL;
        this.specialTokensMapURL = specialTokensMapURL;
        this.tokenizerConfigURL = tokenizerConfigURL;

        try {
            self.postMessage({
                type: "loading",
                message: "Starting file downloads",
            });
            this.config = await fetchArrayBuffer(
                configURL,
                createDownloadHandler(configURL.split("/").pop()),
            );

            this.model = await fetchArrayBuffer(
                modelURL,
                createDownloadHandler(modelURL.split("/").pop()),
            );

            this.vocab = await fetchArrayBuffer(
                vocabURL,
                createDownloadHandler(vocabURL.split("/").pop()),
            );
            this.tokenizerConfig = await fetchArrayBuffer(
                tokenizerConfigURL,
                createDownloadHandler(tokenizerConfigURL.split("/").pop()),
            );
            this.specialTokensMap = await fetchArrayBuffer(
                specialTokensMapURL,
                createDownloadHandler(specialTokensMapURL.split("/").pop()),
            );

            self.postMessage({
                type: "loading",
                message: "All files downloaded successfully",
            });
        } catch (error) {
            self.postMessage({
                type: "error",
                message: `Failed to load files: ${error.message}`,
            });
            throw error;
        }
    }

    /**
       * Initializes and loads the BERT/ModernBert model.
          * @param {string} modelName
       * @returns {Promise<ModernBertPredictor>} - The loaded model instance.
      
       */
    async initializeModel(modelName) {
        if (!Model.loadedModel) {
            await init();
            self.postMessage({ type: "loading", message: "Loading Model" });
            const files = getModel(modelName);
            await this.loadFiles(files);
            Model.loadedModel = new ModernBertPredictor(
                // @ts-ignore
                this.config,
                this.model,
                this.vocab,
                this.tokenizerConfig,
                this.specialTokensMap,
            );

            self.postMessage({
                type: "ready",
                message: "Model Loaded",
                modelName: modelName,
            });
        } else {
            self.postMessage({
                type: "ready",
                message: "Model Already Loaded",
            });
        }
        return Model.loadedModel;
    }

    /**
     * Disposes of the loaded model to free memory.
     */
    dispose() {
        if (Model.loadedModel) {
            Model.loadedModel = undefined;
            self.postMessage({ type: "disposed", message: "Model unloaded" });
        }
    }

    /**
     * @param {object} event - The message event
     * @param {object} event.data
     * @param {'load' | 'infer' | 'dispose'} event.data.type
     * @param {LoadData | InferData} event.data.data The data attached to the type of message
     *
     * @typedef {Object} LoadData
     * @property {string} modelName - The name of the model being loaded.
     * @typedef {Object} InferData
     *
     * @property {Array<string>} text - An array of text
     * @property {Array<string>} hypothesis - An array of labels/hypothesis/classes
     * @property {boolean} multiLabel - Whether to use multiLabel prediction
     *
     */
    async handleMessage(event) {
        const { type, data } = event.data;
        switch (type) {
            case "load":
                const loadData = /** @type {LoadData} */ (data);
                await this.initializeModel(loadData.modelName);
                break;

            case "infer":
                if (!Model.loadedModel) {
                    self.postMessage({
                        type: "error",
                        message: "Model not loaded",
                    });
                    return;
                }
                try {
                    const inferData = /** @type {InferData} */ (data);
                    const result = Model.loadedModel.predict(
                        inferData.text[0],
                        inferData.hypothesis,
                        inferData.multiLabel,
                    );
                    let rawMap = result.to_json();
                    self.postMessage({
                        type: "result",
                        message: {
                            probs: rawMap.get("probs"),
                            labels: rawMap.get("labels"),
                        },
                    });
                } catch (error) {
                    self.postMessage({
                        type: "error",
                        message: `Inference failed: ${error.message}`,
                    });
                }
                break;

            case "dispose":
                this.dispose();
                break;

            default:
                self.postMessage({
                    type: "error",
                    message: `Unknown message type: ${type}`,
                });
        }
    }
}

/**
 * @param {string} modelName
 * @typedef {{configURL: string, modelURL: string, vocabURL: string, tokenizerConfigURL: string, specialTokensMapURL: string}} modelFiles
 * @returns {modelFiles}
 */
function getModel(modelName) {
    const models = {
        MoritzLaurer_ModernBERT_base_zeroshot_v2_0: {
            configURL:
                "https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0/resolve/main/config.json",
            modelURL:
                "https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0/resolve/main/model.safetensors",
            vocabURL:
                "https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0/resolve/main/tokenizer.json",
            tokenizerConfigURL:
                "https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0/resolve/main/tokenizer_config.json",
            specialTokensMapURL:
                "https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0/resolve/main/special_tokens_map.json",
        },
    };
    return models[modelName];
}

(async () => {
    try {
        const modelInstance = new Model();
        /**
         * @param {MessageEvent} event - The message event
         */
        self.onmessage = (event) => modelInstance.handleMessage(event);

        self.postMessage({
            type: "initialized",
            message: "Worker initialized",
        });
    } catch (error) {
        self.postMessage({
            type: "error",
            message: `Initialization failed: ${error.message}`,
        });
    }
})();
