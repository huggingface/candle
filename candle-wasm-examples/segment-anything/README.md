## Running Segment Anything Example

Here, we provide an example showing how to run the Segment Anything model in the
browser.

### Vanilla JS and WebWorkers

To build and test the UI made in Vanilla JS and WebWorkers, first we need to build the WASM library:

```bash
sh build-lib.sh
```

This will bundle the library under `./build` and we can import it inside our WebWorker like a normal JS module:

```js
import init, { Model } from "./build/m.js";
```

The full example can be found under `./lib-example.html`. All needed assets are fetched from the web, so no need to download anything.
Finally, you can preview the example by running a local HTTP server. For example:

```bash
python -m http.server
```

Then open `http://localhost:8000/lib-example.html` in your browser.
