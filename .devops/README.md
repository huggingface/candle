# Vulkan development container

A container for building, testing, and debugging candle with the Vulkan
backend. The container provides the full toolchain (Rust, Vulkan loader and
Mesa Vulkan drivers, Slang shader compiler) plus the debugging and profiling
tools for the Rust code, the Slang shaders, and the compiled SPIR-V, so the
full test suite and the examples can run without any host-side Vulkan or
Rust setup beyond `docker`.

All commands use the `docker` CLI; they work the same on hosts where the
docker runtime is provided by Podman.

## Layout

| File | Purpose |
| --- | --- |
| `vulkan.Dockerfile` | Builds the `candle-vulkan-dev` image from `ubuntu:latest` (multi-stage). |
| `build-vulkan-dev.sh` | Builds the image. |
| `run-vulkan-dev.sh` | Runs the image with GPU access and the usual mounts. |
| `build-install-slang.sh` | Builds the shader-slang programs (static) from a fresh clone and installs them; used by the Dockerfile (into `/usr/local/bin`) and directly on the host (into `$HOME/.local/bin`). |

## Image contents

Base: `ubuntu:latest` (26.04 "resolute" at the time of writing). The image is
built in two stages: a builder compiles Slang, vkpeak, and the Rust profiling
tools from source, and only the resulting binaries are copied into the final
image, so the toolchains and build trees do not bloat it.

Build and run dependencies:

| Package | Provides |
| --- | --- |
| `build-essential`, `cmake`, `pkg-config`, `git`, `curl`, `wget`, `ca-certificates`, `libncurses-dev`, `libxml2-dev` | Core build tools. The `-dev` packages pull in the runtime libraries (`libncurses-dev` → `libtinfo6`, `libxml2-dev` → `libxml2-16` on 26.04), so the image does not depend on the exact versioned runtime package names. |
| `libvulkan-dev` | Vulkan loader (installs `libvulkan1`) and development headers. |
| `vulkan-tools` | `vulkaninfo` and friends. |
| `mesa-vulkan-drivers` | Mesa Vulkan ICD drivers: AMD (radv), Intel (ANV/HASVK), and the llvmpipe software driver. NVIDIA GPUs are not covered by these drivers. |
| `vulkan-validationlayers` | Validation layers, including the printf layer used by the `test_vulkan_shader_printf` test. |
| Slang (source build) | The Slang shader compiler (`/usr/local/bin/slangc`), used by `candle-vulkan-kernels/build.rs` to compile the `.slang` shaders to SPIR-V. Built by `build-install-slang.sh` from a fresh clone pinned by the `SLANG_VERSION` Dockerfile ARG (currently `v2026.21`), with `-DSLANG_LIB_TYPE=STATIC` so the installed programs are mostly self-contained (only system libraries remain as dynamic dependencies). Ubuntu's `slang-compiler` package (2026.1.1) is too old: the native FP8 kernels use the `FloatE4M3` type, which only exists in newer Slang versions. Bump `SLANG_VERSION` when the shaders require a newer Slang feature. |
| Rust (rustup, stable) | Installed under `/usr/local/cargo` / `/usr/local/rustup`. |

Debugging and profiling tools:

| Tool | Purpose |
| --- | --- |
| `spirv-tools` | `spirv-dis` (disassemble), `spirv-val` (validate), and `spirv-opt` (optimize/inspect) for the SPIR-V binaries `build.rs` produces. |
| `apitrace` | Vulkan/OpenGL call tracing (`apitrace` CLI). The `qapitrace` GUI needs a display. |
| `gdb` | Native/Rust debugger. |
| `flamegraph` (cargo) | CPU flame graphs. On Linux this shells out to a host `perf` binary, which must match the host kernel and is not installed in the image (see Notes). |
| `samply` (cargo) | Sampling profiler (self-contained; no `perf` binary needed). |
| `hyperfine` (cargo) | Command-line benchmarking. |
| `vkpeak` (source build) | Vulkan compute peak-throughput benchmark, built from `https://github.com/nihui/vkpeak` (with its `ncnn` submodule). |

RenderDoc is intentionally omitted: it is not packaged for Ubuntu 26.04, and
its primary value (the `qrenderdoc` GUI) is unusable in a headless
container. `spirv-tools` (SPIR-V inspection/validation) and `apitrace` (API
tracing) cover the debugging workflows RenderDoc would otherwise provide for
a compute-only backend.

Note: LunarG's Ubuntu APT repository (`packages.lunarg.com/vulkan/ubuntu`)
is no longer available (its URLs return 404), so the Vulkan toolchain comes
from Ubuntu's own packages, the Slang compiler and vkpeak are built from
source, and the Rust profiling tools are installed with `cargo install`.

## Build

```sh
.devops/build-vulkan-dev.sh
```

Tags the image `candle-vulkan-dev` (override with `CANDLE_VULKAN_IMAGE`).
The build context is the `.devops` directory only, since the Dockerfile
copies nothing from the repository.

## Run

```sh
# Interactive shell (the checkout is mounted at /mnt/workspace):
.devops/run-vulkan-dev.sh

# Run a command inside the container:
.devops/run-vulkan-dev.sh -- cargo check --features vulkan -p candle-vulkan-kernels

# Select the vulkan device for the test suite (a device index; 0 on a
# single-GPU host):
CANDLE_VULKAN_TEST_GPU=1 .devops/run-vulkan-dev.sh -- \
    cargo test --features vulkan --workspace --no-fail-fast
```

The run script passes the host GPU through (`--device /dev/dri`, which
assumes a Linux host), mounts the candle checkout (located automatically
from the script, i.e. the parent of `.devops`; override with `CANDLE_DIR`)
at `/mnt/workspace`, and mounts the host's Hugging Face cache at
`/root/.cache/huggingface` so models and the HF token do not have to be
re-downloaded. It also grants `SYS_PTRACE` and `PERFMON` so the debugger and
profilers can inspect the processes they launch.

Host environment variables that are set are forwarded into the container:
`CANDLE_BACKEND`, `CANDLE_VULKAN_TEST_GPU`, `CANDLE_VULKAN_SLANG_OPT`,
`MESA_VK_DEVICE_SELECT`, and `HF_TOKEN`. On Linux hosts, when the caller
is not a login shell, an unset `MESA_VK_DEVICE_SELECT` falls back to the
value in `/etc/environment`, so the container's device ordering matches
the host's when the host pins a GPU there. Without such a pin the
container's device ordering is the driver's default enumeration, which
may differ from the host's (for example a software or secondary device
may come first), so `CANDLE_VULKAN_TEST_GPU` indices must not be copied
blindly between host and container.

Equivalent raw invocation (where `CANDLE_DIR` is the checkout path):

```sh
docker run -it \
  --device /dev/dri \
  --cap-add=SYS_PTRACE \
  --cap-add=PERFMON \
  --volume "${CANDLE_DIR}:/mnt/workspace" \
  candle-vulkan-dev
```

## Verification

```sh
# Verify the Slang shader compiler:
slangc -version

# Verify Vulkan device bindings (the host's GPU should be listed):
vulkaninfo --summary

# Verify the debug/profiling tools:
spirv-dis --version
apitrace --version
hyperfine --version
samply --version
vkpeak --help
```

## Running tests

From inside the container (working directory `/mnt/workspace`), the same
checks apply as on the host. First, determine which devices the container
sees and which are real GPUs. Never target a
`PHYSICAL_DEVICE_TYPE_CPU` device (llvmpipe): it is only a CPU fallback so
the display keeps working when the GPU is unavailable, and anything that can
run there runs faster on candle's own CPU backend.

```sh
vulkaninfo --summary
```

Run the workspace test suite once per **real GPU** device index, selecting
the device with `CANDLE_VULKAN_TEST_GPU` (on a single-GPU host, only the
device 0 run is needed). The clippy, bench, and shader-printf checks do not
depend on the device index:

```sh
cargo clippy --all-targets --features vulkan
CANDLE_VULKAN_TEST_GPU=0 cargo test --features vulkan --workspace --no-fail-fast
CANDLE_VULKAN_TEST_GPU=1 cargo test --features vulkan --workspace --no-fail-fast  # second GPU only
cargo bench --features vulkan --profile release -- --test
VK_LAYER_PRINTF_ENABLE=1 VK_LAYER_PRINTF_TO_STDOUT=1 \
    cargo test --features vulkan --test vulkan_integration_tests \
    -- test_vulkan_shader_printf --exact
```

`CANDLE_VULKAN_TEST_GPU` selects the device index; the run script forwards
it from the host, so the suite can also be run without entering the
container:

```sh
CANDLE_VULKAN_TEST_GPU=1 .devops/run-vulkan-dev.sh -- \
    cargo test --features vulkan --workspace --no-fail-fast
```

## Notes

- The `target/` directory is shared with the host through the mount. The
  first build inside the container may recompile the workspace (different
  toolchain path/glibc), later runs are incremental.
- The Mesa ICD drivers in the image cover AMD and Intel GPUs (and
  llvmpipe). A host with an NVIDIA GPU needs a different GPU passthrough
  configuration (for example the nvidia container runtime) instead of
  `--device /dev/dri`.
- `flamegraph` on Linux drives the host `perf` tool, which is
  kernel-version-specific and is not installed in the image. To use it,
  mount or install a `perf` binary matching the host kernel (the granted
  `PERFMON` capability then lets it sample). `samply` is the recommended
  profiler inside the container because it is self-contained.
- `apitrace`'s GUI front-end (`qapitrace`) is installed but needs a display
  (X11 forwarding) to open; the `apitrace` CLI works headless.
- Large model runs (e.g. the `muse-glimmer` example) should be checked
  against any other GPU load on the host.