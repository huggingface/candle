# Vulkan development container for candle.
#
# Base image: ubuntu:latest (26.04 "resolute" at the time of writing).
#
# Note on the Vulkan toolchain: LunarG's Ubuntu APT repository
# (packages.lunarg.com/vulkan/ubuntu) is no longer available (its URLs
# return 404), so this image uses Ubuntu's own packages, which provide
# everything needed to build and run candle with the Vulkan backend:
#   - libvulkan-dev:           Vulkan loader (installs libvulkan1) and
#                              development headers
#   - vulkan-tools:            vulkaninfo and friends
#   - mesa-vulkan-drivers:     Mesa Vulkan ICD drivers (AMD, Intel,
#                              llvmpipe software; no NVIDIA)
#   - vulkan-validationlayers: validation layers (incl. the printf layer)
#   - Slang:                    the Slang shader compiler (slangc), built
#                               from source (stage "builder"), used by
#                               candle-vulkan-kernels/build.rs to compile
#                               the .slang shaders to SPIR-V
#
# The image is multi-stage: a "builder" compiles Slang, vkpeak, and the
# Rust profiling tools from source, and only the resulting binaries are
# copied into the final image, so the toolchains and build trees do not
# bloat it.
#
# Build with .devops/build-vulkan-dev.sh, run with .devops/run-vulkan-dev.sh.

# ---------------------------------------------------------------------------
# Stage 1: builder. Compiles everything that is built from source.
# ---------------------------------------------------------------------------
FROM ubuntu:latest AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Build tools, Vulkan headers, and glslang (glslangValidator) for the
# vkpeak/ncnn shader compilation, plus libx11-dev for Slang's SPIR-V/Vulkan
# codegen path.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    pkg-config \
    git \
    wget \
    python3 \
    libncurses-dev \
    libxml2-dev \
    libx11-dev \
    libvulkan-dev \
    glslang-tools \
    && rm -rf /var/lib/apt/lists/*

# vkpeak, the Vulkan compute peak-throughput benchmark, built from source
# (with its ncnn submodule, which compiles ncnn's Vulkan shaders via
# glslangValidator). The resulting binary links ncnn statically.
RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/nihui/vkpeak.git /tmp/vkpeak \
    && cmake -S /tmp/vkpeak -B /tmp/vkpeak/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/vkpeak/build -j "$(nproc)" \
    && cp /tmp/vkpeak/build/vkpeak /usr/local/bin/vkpeak \
    && rm -rf /tmp/vkpeak

# Slang shader compiler, built from source by build-install-slang.sh.
# Ubuntu's slang-compiler package (2026.1.1) is too old for the candle
# shaders: the native FP8 kernels use the FloatE4M3 type, which only exists
# in newer Slang versions. The script clones a pinned release under $HOME,
# builds it with SLANG_LIB_TYPE=STATIC (so the programs are mostly
# self-contained), installs them under /usr/local/bin (and the dlopen-ed
# downstream-compiler plugins under /usr/local/lib), and removes the source
# and build trees (--clean). Bump SLANG_VERSION when the shaders require a
# newer Slang feature.
ARG SLANG_VERSION=v2026.21
COPY build-install-slang.sh /tmp/build-install-slang.sh
RUN sh /tmp/build-install-slang.sh --version "${SLANG_VERSION}" --install-dir /usr/local/bin --clean

# Rust toolchain, used here only to compile the profiling/benchmark tools.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
# sh.rustup.rs is the installer script; plain rustup.rs serves an HTML
# landing page and cannot be piped to a shell.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Install to a private root so only the binaries are copied into the final
# image (and the Rust toolchain itself is not).
RUN cargo install --root /usr/local/cargo-tools flamegraph samply hyperfine

# ---------------------------------------------------------------------------
# Stage 2: final image. Runtime dependencies + Rust, plus the binaries built
# in stage 1.
# ---------------------------------------------------------------------------
FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

# 1. Core build tools.
# The -dev packages are used instead of the versioned runtime libraries
# (libncurses-dev pulls in the libtinfo6 runtime, libxml2-dev pulls in
# libxml2-16 on 26.04) so the image does not depend on the exact
# versioned runtime package names. libx11-dev (which pulls libxcb1-dev)
# is needed to link the workspace: vulkano, the Vulkan wrapper candle
# uses, depends on x11rb/x11-dl for window-system integration, so tests
# and examples link against libxcb.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    pkg-config \
    git \
    wget \
    python3 \
    libncurses-dev \
    libxml2-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Vulkan loader, tools, Mesa ICD drivers (AMD, Intel, software) and
# validation layers. (libvulkan-dev pulls in the libvulkan1 loader.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    vulkan-validationlayers \
    && rm -rf /var/lib/apt/lists/*

# 3. Debugging and profiling tools. (RenderDoc is intentionally omitted:
# it is not packaged for Ubuntu 26.04, and its value is the qrenderdoc GUI,
# which is unusable in a headless container; apitrace and spirv-tools cover
# tracing and SPIR-V inspection.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    spirv-tools \
    apitrace \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# 4. Rust toolchain (needed to build candle inside the container).
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 5. Copy the source-built artifacts from the builder stage.
COPY --from=builder /usr/local/bin/slang* /usr/local/bin/
COPY --from=builder /usr/local/lib/libslang-*.so /usr/local/lib/
COPY --from=builder /usr/local/bin/vkpeak /usr/local/bin/
COPY --from=builder /usr/local/cargo-tools/bin/ /usr/local/bin/

WORKDIR /mnt/workspace
CMD ["/bin/bash"]