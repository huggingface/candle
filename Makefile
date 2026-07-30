.PHONY: clean-ptx clean test all \
        check check-cuda check-rocm check-all \
        fmt fmt-check clippy clippy-rocm \
        test-rocm test-rocm-core test-rocm-nn test-rocm-suite \
        rocm-info rocm-cache-clean rocm-shim-test

CARGO ?= cargo
export CARGO_BUILD_JOBS ?= 4

# ROCm test tuning. Kernels are compiled by hipcc on first use and cached on
# disk. That cache is now locked per entry, so concurrent compilers are safe;
# the suites still default to one thread so that GPU memory use and the
# attribution of a failure stay predictable. Override with ROCM_TEST_THREADS.
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

# Compiles every shared candle-kernels source with hipcc and checks the two
# pieces of hand-written device code in the shim (16-bit atomicAdd, the *_sync
# shuffle macros) on the actual GPU.
ROCM_ARCH ?= $(shell rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-z]*')
rocm-shim-test:
	@shim=candle-rocm-kernels/src/hip_shim; \
	tmp=$$(mktemp -d); trap 'rm -rf $$tmp' EXIT; \
	for f in candle-kernels/src/*.cu; do \
	  case $$f in *mmvq_gguf.cu) continue;; esac; \
	  n=$$(basename $$f .cu); printf '%-12s' "$$n"; \
	  hipcc --genco --offload-arch=$(ROCM_ARCH) -O3 -std=c++17 -D__CUDA_ARCH__=800 \
	    -include $$shim/hip_compat.h -I $$shim -I candle-kernels/src \
	    -o $$tmp/$$n.hsaco $$f 2>$$tmp/$$n.err \
	    && echo "compiled" || { echo "FAILED"; sed -n '1,20p' $$tmp/$$n.err; exit 1; }; \
	done; \
	hipcc --offload-arch=$(ROCM_ARCH) -std=c++17 -I $$shim -o $$tmp/shim_test $$shim/shim_test.hip \
	  && $$tmp/shim_test

rocm-info:
	rocminfo | grep -E '^\s*(Name|Marketing Name|Uuid):' || true
	hipcc --version | head -3

# Compiled HIP code objects are cached here. The cache key covers the sources,
# the shim headers, the compile flags and the toolchain, so editing any of them
# already invalidates the entry; this target is for clearing a cache some crash
# left behind. CANDLE_ROCM_FORCE_RECOMPILE=1 does the same for a single run.
ROCM_CACHE_DIR ?= $(if $(CANDLE_ROCM_CACHE_DIR),$(CANDLE_ROCM_CACHE_DIR),$(HOME)/.cache/candle-rocm)
rocm-cache-clean:
	rm -rf $(ROCM_CACHE_DIR)

all: fmt-check clippy test
