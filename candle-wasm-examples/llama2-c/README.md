## Running [llama2.c](https://github.com/karpathy/llama2.c) Examples

Here, we provide two examples of how to run Whisper using a Candle-compiled WASM binary and runtimes.

### Pure Rust UI

To build and test the UI made in Rust you will need [Trunk](https://trunkrs.dev/#install)
From the `candle-wasm-examples/whisper` directory run:

Download assets:

```bash
# Model and tokenizer
wget -c https://huggingface.co/spaces/lmz/candle-whisper/resolve/main/mel_filters.safetensors
wget -c https://huggingface.co/spaces/lmz/candle-whisper/resolve/main/tiny.en.safetensors
wget -c https://huggingface.co/spaces/lmz/candle-whisper/resolve/main/tokenizer.en.json


# Audio samples
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_gb0.wav -O gb0.wav
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_a13.wav -O a13.wav
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_gb1.wav -O gb1.wav
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_hp0.wav -O hp0.wav
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_jfk.wav -O jfk.wav
wget -c https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_mm0.wav -O mm0.wav

```

Run hot reload server:

```bash
trunk serve --release --public-url / --port 8080
```

### Vanilla JS and WebWorkers

To build and test the UI made in Vanilla JS and WebWorkers, first we need to build the WASM library:

```bash
sh build-lib.sh
```

This will bundle the library under `./build` and we can import it inside our WebWorker like a normal JS module:

```js
import init, { Decoder } from "./build/m.js";
```

The full example can be found under `./lib-example.html`. All needed assets are fetched from the web, so no need to download anything.
Finally, you can preview the example by running a local HTTP server. For example:

```bash
python -m http.server
```

Then open `http://localhost:8000/lib-example.html` in your browser.
