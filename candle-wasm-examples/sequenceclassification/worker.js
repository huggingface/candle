/**
 * @typedef {Uint8Array} U8Array
 * @description An array of unsigned 8-bit integers (0-255).
 */
import init, { BertPredictor, ModernBertPredictor } from "./build/m.js";

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
    /** @type {BertPredictor | ModernBertPredictor | undefined} */
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
       * @returns {Promise<BertPredictor>} - The loaded model instance.
      
       */
    async initializeModel(modelName) {
        if (!Model.loadedModel) {
            await init();
            self.postMessage({ type: "loading", message: "Loading Model" });
            const files = getModel(modelName);
            await this.loadFiles(files);
            if (modelName.toLowerCase().includes("modernbert")) {
                Model.loadedModel = new ModernBertPredictor(
                    // @ts-ignore
                    this.config,
                    this.model,
                    this.vocab,
                    this.tokenizerConfig,
                    this.specialTokensMap,
                );
            } else {
                let numLabels = null;
                if (modelName == "textattack_bert_base_uncased_yelp_polarity") {
                    numLabels = 2;
                }

                Model.loadedModel = new BertPredictor(
                    // @ts-ignore
                    this.config,
                    this.model,
                    this.vocab,
                    this.tokenizerConfig,
                    this.specialTokensMap,
                    numLabels,
                );
            }
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
     * @param {MessageEvent} event - The message event
     */
    async handleMessage(event) {
        const { type, data } = event.data;
        switch (type) {
            case "load":
                await this.initializeModel(data.modelName);
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
                    const result = Model.loadedModel.predict(data.text);
                    let rawMap = result.to_json();
                    const id2label = rawMap.get("id2label");
                    self.postMessage({
                        type: "result",
                        message: {
                            probs: rawMap.get("probs"),
                            labels: id2label
                                ? [...id2label.values()]
                                : rawMap.get("labels"),
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
    const MODELS = {
        textattack_bert_base_uncased_yelp_polarity: {
            configURL:
                "https://huggingface.co/textattack/bert-base-uncased-yelp-polarity/resolve/refs%2Fpr%2F1/config.json",
            modelURL:
                "https://huggingface.co/textattack/bert-base-uncased-yelp-polarity/resolve/refs%2Fpr%2F1/model.safetensors",
            vocabURL:
                "https://huggingface.co/textattack/bert-base-uncased-yelp-polarity/resolve/refs%2Fpr%2F1/vocab.txt",
            tokenizerConfigURL:
                "https://huggingface.co/textattack/bert-base-uncased-yelp-polarity/resolve/main/tokenizer_config.json",
            specialTokensMapURL:
                "https://huggingface.co/textattack/bert-base-uncased-yelp-polarity/resolve/main/special_tokens_map.json",
        },
        nlptown_bert_base_multilingual_uncased_sentiment: {
            configURL:
                "https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/resolve/main/config.json",
            modelURL:
                "https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/resolve/main/model.safetensors",
            vocabURL:
                "https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/resolve/main/vocab.txt",
            tokenizerConfigURL:
                "https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/resolve/main/tokenizer_config.json",
            specialTokensMapURL:
                "https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/resolve/main/special_tokens_map.json",
        },
        cirimus_modernbert_base_go_emotions: {
            configURL:
                "https://huggingface.co/cirimus/modernbert-base-go-emotions/resolve/main/config.json",
            modelURL:
                "https://huggingface.co/cirimus/modernbert-base-go-emotions/resolve/main/model.safetensors",
            vocabURL:
                "https://huggingface.co/cirimus/modernbert-base-go-emotions/resolve/main/tokenizer.json",
            tokenizerConfigURL:
                "https://huggingface.co/cirimus/modernbert-base-go-emotions/resolve/main/tokenizer_config.json",
            specialTokensMapURL:
                "https://huggingface.co/cirimus/modernbert-base-go-emotions/resolve/main/special_tokens_map.json",
        },
        clapAI_modernBERT_base_multilingual_sentiment: {
            configURL:
                "https://huggingface.co/clapAI/modernBERT-base-multilingual-sentiment/resolve/main/config.json",
            modelURL:
                "https://huggingface.co/clapAI/modernBERT-base-multilingual-sentiment/resolve/main/model.safetensors",
            vocabURL:
                "https://huggingface.co/clapAI/modernBERT-base-multilingual-sentiment/resolve/main/tokenizer.json",
            tokenizerConfigURL:
                "https://huggingface.co/clapAI/modernBERT-base-multilingual-sentiment/resolve/main/tokenizer_config.json",
            specialTokensMapURL:
                "https://huggingface.co/clapAI/modernBERT-base-multilingual-sentiment/resolve/main/special_tokens_map.json",
        },
    };
    return MODELS[modelName];
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
