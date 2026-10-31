# Vulkan development container

A container for building and testing candle with the Vulkan backend. The
container provides the full toolchain (Rust, Vulkan loader and Mesa
Vulkan drivers, Slang shader compiler) so the usual test battery and the
examples can run without any host-side Vulkan or Rust setup beyond `docker`.

All commands use the `docker` CLI; they work the same on hosts where the
docker runtime is provided by Podman.

## Layout

| File | Purpose |
| --- | --- |
| `vulkan.Dockerfile` | Builds the `candle-vulkan-dev` image from `ubuntu:latest`. |
| `build-vulkan-dev.sh` | Builds the image. |
| `run-vulkan-dev.sh` | Runs the image with GPU access and the usual mounts. |

## Image contents

Base: `ubuntu:latest` (26.04 "resolute" at the time of writing).

| Package | Provides |
| --- | --- |
| `build-essential`, `cmake`, `pkg-config`, `git`, `curl`, `wget`, `ca-certificates`, `libtinfo6`, `libxml2-16` | Core build tools (`libxml2-16` is the versioned runtime name of `libxml2` on 26.04). |
| `libvulkan1` | Vulkan loader. |
| `vulkan-tools` | `vulkaninfo` and friends. |
| `mesa-vulkan-drivers` | Mesa Vulkan ICD drivers: AMD (radv), Intel (ANV/HASVK), and the llvmpipe software driver. NVIDIA GPUs are not covered by these drivers. |
| `vulkan-validationlayers` | Validation layers, including the printf layer used by the `test_vulkan_shader_printf` test. |
| Slang (source build) | The Slang shader compiler (`/usr/local/bin/slangc`, installed under `/opt/slang`), used by `candle-vulkan-kernels/build.rs` to compile the `.slang` shaders to SPIR-V. Built from source pinned by the `SLANG_VERSION` Dockerfile ARG (currently `v2026.18.2`); the source tree is cloned and built under `/root/slang` because a full build needs a few gigabytes and `/tmp` may be too small on some systems. Ubuntu's `slang-compiler` package (2026.1.1) is too old: the native FP8 kernels use the `FloatE4M3` type, which only exists in newer Slang versions. Bump `SLANG_VERSION` to match the host's `slangc -version` when the host toolchain changes. |
| Rust (rustup, stable) | Installed under `/usr/local/cargo` / `/usr/local/rustup`. |

Note: LunarG's Ubuntu APT repository (`packages.lunarg.com/vulkan/ubuntu`)
is no longer available (its URLs return 404), so the Vulkan toolchain comes
from Ubuntu's own packages and the Slang compiler is built from source
(the packaged version is too old for the candle shaders).

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
assumes a Linux host), mounts `$HOME/workspace/candle` (override with
`CANDLE_DIR`) at `/mnt/workspace`, and mounts the host's Hugging Face
cache at `/root/.cache/huggingface` so models and the HF token do not
have to be re-downloaded.

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

Equivalent raw invocation:

```sh
docker run -it \
  --device /dev/dri \
  --volume "${HOME}/workspace/candle:/mnt/workspace" \
  candle-vulkan-dev
```

## Verification

```sh
# Verify the Slang shader compiler:
slangc -version

# Verify Vulkan device bindings (the host's GPU should be listed):
vulkaninfo --summary
```

## Usual test battery

From inside the container (working directory `/mnt/workspace`), the same
checks as on the host apply. On a host with multiple vulkan devices, run
the workspace test suite once per device; on a single-device host only
the first test run is needed:

```sh
cargo clippy --all-targets --features vulkan
CANDLE_VULKAN_TEST_GPU=0 cargo test --features vulkan --workspace --no-fail-fast
CANDLE_VULKAN_TEST_GPU=1 cargo test --features vulkan --workspace --no-fail-fast
cargo bench --features vulkan --profile release -- --test
VK_LAYER_PRINTF_ENABLE=1 VK_LAYER_PRINTF_TO_STDOUT=1 \
    cargo test --features vulkan --test vulkan_integration_tests \
    -- test_vulkan_shader_printf --exact
```

`CANDLE_VULKAN_TEST_GPU` selects the device index; the run script
forwards it from the host, so it can also be set outside:

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
- Large model runs (e.g. the `muse-glimmer` example) should be checked
  against any other GPU load on the host.
