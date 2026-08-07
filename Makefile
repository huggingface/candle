.PHONY: clean-ptx clean test all \
        check check-cuda check-rocm check-all \
        fmt fmt-check clippy clippy-rocm \
        test-rocm test-rocm-core test-rocm-nn test-rocm-suite test-rocm-ug \
        rocm-info rocm-cache-clean rocm-shim-test

CARGO ?= cargo
export CARGO_BUILD_JOBS ?= 4

# ROCm test tuning. Kernels are compiled by hipcc on first use and cached on
# disk. That cache is now locked per entry, so concurrent compilers are safe;
# the suites still default to one thread so that GPU memory use and the
# attribution of a failure stay predictable. Override with ROCM_TEST_THREADS.
# Plain `rocm`, not `_rocm`: the integration suites suffix their GPU tests
# `..._rocm`, but the backend's own module tests are named for what they check
# and only carry `rocm` in their module path (`quantized::rocm::tests_mmvq::…`).
# The narrower filter silently skipped every one of them.
ROCM_FILTER ?= rocm
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

# Mirrors `.github/workflows/rust-ci.yml`'s clippy job exactly, `--benches`
# included. Dropping any of those flags means passing here and failing CI.
clippy:
	$(CARGO) clippy --workspace --tests --examples --benches -- -D warnings

# Same flags as `clippy` above, minus the members that have no ROCm code.
clippy-rocm:
	$(CARGO) clippy -p candle-core -p candle-nn --features rocm --tests --benches -- -D warnings

test:
	$(CARGO) test --workspace

test-rocm-core:
	$(CARGO) test -p candle-core --features rocm -- $(ROCM_FILTER) --test-threads=$(ROCM_TEST_THREADS)

test-rocm-nn:
	$(CARGO) test -p candle-nn --features rocm -- $(ROCM_FILTER) --test-threads=$(ROCM_TEST_THREADS)

test-rocm: test-rocm-core test-rocm-nn

# The `ug` micro-kernel path. Separate from test-rocm-core because `ug` is not
# part of the `rocm` feature set, so the ROCm suites above never build UgIOp1.
# The filter is `ug` rather than $(ROCM_FILTER): the test is named for the op.
test-rocm-ug:
	$(CARGO) test -p candle-core --features "rocm ug" -- ug --test-threads=$(ROCM_TEST_THREADS)

# Run one candle-core suite against the GPU, e.g. make test-rocm-suite SUITE=conv_tests
test-rocm-suite:
	$(CARGO) test -p candle-core --features rocm --test $(SUITE) -- $(ROCM_FILTER) --test-threads=$(ROCM_TEST_THREADS)

# Compiles every shared candle-kernels source with hipcc and checks the
# hand-written device code in the shim (16-bit atomicAdd, the *_sync shuffle
# macros, __dp4a) on the actual GPU.
ROCM_ARCH ?= $(shell rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-z]*')
# Read from COMPILE_FLAGS rather than repeated here: compiling the shim test at
# a different __CUDA_ARCH__ than the runtime compiler uses would test a
# different set of kernels than the one that actually ships.
ROCM_ARCH_DEFINE = $(shell grep -om1 '\-D__CUDA_ARCH__=[0-9]*' candle-rocm-kernels/src/compile/cache.rs)
rocm-shim-test:
	@shim=candle-rocm-kernels/src/hip_shim; \
	test -n "$(ROCM_ARCH_DEFINE)" || { echo "could not read __CUDA_ARCH__ from cache.rs"; exit 1; }; \
	tmp=$$(mktemp -d); trap 'rm -rf $$tmp' EXIT; \
	for f in candle-kernels/src/*.cu; do \
	  case $$f in *mmvq_gguf.cu) continue;; esac; \
	  n=$$(basename $$f .cu); printf '%-12s' "$$n"; \
	  hipcc --genco --offload-arch=$(ROCM_ARCH) -O3 -std=c++17 $(ROCM_ARCH_DEFINE) \
	    -include $$shim/hip_compat.h -I $$shim -I candle-kernels/src \
	    -o $$tmp/$$n.hsaco $$f 2>$$tmp/$$n.err \
	    && echo "compiled" || { echo "FAILED"; sed -n '1,20p' $$tmp/$$n.err; exit 1; }; \
	done; \
	hipcc --offload-arch=$(ROCM_ARCH) -std=c++17 -I $$shim -o $$tmp/shim_test $$shim/shim_test.hip \
	  && $$tmp/shim_test || exit 1; \
	rocm_inc=$$(hipconfig --rocmpath 2>/dev/null || echo /opt/rocm)/include/hip/amd_detail; \
	if grep -lq atomicAdd $$rocm_inc/amd_hip_fp16.h $$rocm_inc/amd_hip_bf16.h 2>/dev/null; then \
	  echo "coexist test skipped: HIP headers define 16-bit atomicAdd (shim_test above covered coexistence)"; \
	else \
	  hipcc --offload-arch=$(ROCM_ARCH) -std=c++17 -I $$shim -o $$tmp/shim_coexist_test \
	    $$shim/shim_coexist_test.hip && $$tmp/shim_coexist_test; \
	fi

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
