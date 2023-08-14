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

pyo3-test:
	cargo build --profile=release-with-debug --package candle-pyo3
	python3 candle-pyo3/test.py

all: test
