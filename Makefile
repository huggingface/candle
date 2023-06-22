clean-ptx:
	find target -name "*.ptx" -type f -delete
	echo "" > kernels/src/lib.rs
	touch kernels/build.rs

clean:
	cargo clean

test:
	cargo test

all: test
