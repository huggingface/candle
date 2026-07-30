.PHONY: clean-ptx clean test all \
        check check-cuda check-rocm check-all \
        fmt fmt-check clippy clippy-rocm \
        test-rocm test-rocm-core test-rocm-nn test-rocm-suite \
        rocm-info rocm-cache-clean

CARGO ?= cargo
export CARGO_BUILD_JOBS ?= 4

# ROCm test tuning. Kernels are compiled by hipcc on first use and cached on
# disk; until that cache is race-free the ROCm suites run single-threaded.
ROCM_FILTER ?= _rocm
ROCM_TEST_THREADS ?= 1
ROCM_CRATES := -p candle-core -p candle-nn -p candle-transformers -p candle-examples

clean-ptx:
	find target -name "*.ptx" -type f -delete
	echo "" > candle-kernels/src/lib.rs
	touch candle-kernels/build.rs
	touch candle-examples/build.rs
	touch candle-flash-attn/build.rs

clean:
	$(CARGO) clean

check:
	$(CARGO) check --workspace

check-cuda:
	$(CARGO) check --workspace --features cuda

check-rocm:
	$(CARGO) check $(ROCM_CRATES) --features rocm
	$(CARGO) check --manifest-path candle-rocm-kernels/Cargo.toml

check-all: check check-cuda check-rocm

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --workspace --tests --examples -- -D warnings

clippy-rocm:
	$(CARGO) clippy -p candle-core -p candle-nn --features rocm --tests -- -D warnings

test:
	$(CARGO) test --workspace

test-rocm-core:
	$(CARGO) test -p candle-core --features rocm -- $(ROCM_FILTER) --test-threads=$(ROCM_TEST_THREADS)

test-rocm-nn:
	$(CARGO) test -p candle-nn --features rocm -- $(ROCM_FILTER) --test-threads=$(ROCM_TEST_THREADS)

test-rocm: test-rocm-core test-rocm-nn

# Run one candle-core suite against the GPU, e.g. make test-rocm-suite SUITE=conv_tests
test-rocm-suite:
	$(CARGO) test -p candle-core --features rocm --test $(SUITE) -- $(ROCM_FILTER) --test-threads=$(ROCM_TEST_THREADS)

rocm-info:
	rocminfo | grep -E '^\s*(Name|Marketing Name|Uuid):' || true
	hipcc --version | head -3

# Compiled HIP code objects are cached here; a stale entry after editing a
# kernel source is the most common bring-up failure.
rocm-cache-clean:
	rm -rf $(HOME)/.cache/candle-rocm

all: fmt-check clippy test
