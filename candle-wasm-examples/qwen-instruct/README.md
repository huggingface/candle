## Running [Qwen2.5-Instruct](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) Examples

Here, we provide examples of how to run [Qwen2.5-Instruct](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) models written in Rust using a Candle-compiled WASM binary and runtime.

### Vanilla JS and WebWorkers

To build and test the UI made in Vanilla JS and WebWorkers, first we need to build the WASM library:

```bash
sh build-lib.sh
```

This will bundle the library under `./build` and w can import it inside our WebWorker like a normal JS module:

```js
import init, { Model } from "./build/m.js";
```

The full example can be found under `./index.html`. All needed assets are fetched from the web, so no need to download anything.
Finally, you can preview the example by running a local HTTP server. For example:

```bash
python -m http.server
```

Then open `http://localhost:8000/index.html` in your browser.