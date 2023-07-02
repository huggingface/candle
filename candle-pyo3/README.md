```
cargo build --release --package candle-pyo3 --no-default-features && cp -f target/release/libcandle.so candle.so
PYTHONPATH=. python3 candle-pyo3/test.py
```
