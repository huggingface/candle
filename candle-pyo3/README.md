From the top level directory run the following for linux.
```
cargo build --profile=release-with-debug --package candle-pyo3 && cp -f ./target/release-with-debug/libcandle.so candle.so
PYTHONPATH=. python3 candle-pyo3/test.py
```bash

  Or for macOS users:
```bash
cargo build --profile=release-with-debug --package candle-pyo3 && cp -f ./target/release-with-debug/libcandle.dylib candle.so
PYTHONPATH=. python3 candle-pyo3/test.py
```
