.PHONY: clean-ptx clean test

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
