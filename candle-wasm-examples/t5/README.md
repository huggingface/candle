## Running T5 with Candle and WASM

Here, we provide two examples of how to run Bert using a Candle-compiled WASM binary and runtime.

### Vanilla JS and WebWorkers

To build and test the UI made in Vanilla JS and WebWorkers, first we need to build the WASM library:

```bash
sh build-lib.sh
```

This will bundle the library under `./build` and we can import it inside our WebWorker like a normal JS module:

```js
import init, { ModelConditionalGeneration, ModelEncoder } from "./build/m.js";
```

For the quantized version, we need to import the quantized module:

```js
import init, { ModelConditionalGeneration, ModelEncoder } from "./build/m-quantized.js";
```

The full example can be found under `./index.html`. All needed assets are fetched from the web, so no need to download anything.
Finally, you can preview the example by running a local HTTP server. For example:

```bash
python -m http.server
```

Then open `http://localhost:8000/index.html` in your browser.
