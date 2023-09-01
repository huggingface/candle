pyo3-test:
	cargo build --profile=release-with-debug --package candle-pyo3
	cp -f ./target/release-with-debug/libcandle.so candle.so
	PYTHONPATH=.:$PYTHONPATH python3 candle-pyo3/test.py

pyo3-test-macos:
	cargo build --profile=release-with-debug --package candle-pyo3
	cp -f ./target/release-with-debug/libcandle.dylib candle.so
	PYTHONPATH=.:$PYTHONPATH python3 candle-pyo3/test.py

.PHONY: pyo3-test pyo3-test-macos clean-ptx clean test

clean-ptx:
	find target -name "*.ptx" -type f -delete
	echo "" > candle-kernels/src/lib.rs
	touch candle-kernels/build.rs
	touch candle-examples/build.rs
	touch candle-flash-attn/build.rs

clean:
	cargo clean

test:
	cargo test

all: test
