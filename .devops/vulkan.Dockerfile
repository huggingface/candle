# Vulkan development container for candle.
#
# Base image: ubuntu:latest (26.04 "resolute" at the time of writing).
#
# Note on the Vulkan toolchain: LunarG's Ubuntu APT repository
# (packages.lunarg.com/vulkan/ubuntu) is no longer available (its URLs
# return 404), so this image uses Ubuntu's own packages, which provide
# everything needed to build and run candle with the Vulkan backend:
#   - libvulkan1:              Vulkan loader
#   - vulkan-tools:            vulkaninfo and friends
#   - mesa-vulkan-drivers:     Mesa Vulkan ICD drivers (AMD, Intel,
#                              llvmpipe software; no NVIDIA)
#   - vulkan-validationlayers: validation layers (incl. the printf layer)
#   - Slang:                    the Slang shader compiler (slangc), built
#                               from source (see step 3), used by
#                               candle-vulkan-kernels/build.rs to compile
#                               the .slang shaders to SPIR-V
#
# Build with .devops/build-vulkan-dev.sh, run with .devops/run-vulkan-dev.sh.

FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

# 1. Core build tools.
# (libxml2-16 is the versioned runtime name of libxml2 on 26.04.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    pkg-config \
    git \
    wget \
    python3 \
    libtinfo6 \
    libxml2-16 \
    && rm -rf /var/lib/apt/lists/*

# 2. Vulkan loader, tools, Mesa ICD drivers (AMD, Intel, software) and
# validation layers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    vulkan-validationlayers \
    && rm -rf /var/lib/apt/lists/*

# 3. Slang shader compiler, built from source.
# Ubuntu's slang-compiler package (2026.1.1) is too old for the candle
# shaders: the native FP8 kernels use the FloatE4M3 type, which only
# exists in newer Slang versions. The pinned version should match the
# host's slangc (`slangc -version`); bump SLANG_VERSION when it changes.
# The source tree is cloned and built under /root (not /tmp) because the
# checkout plus build tree need a few gigabytes and /tmp may be too
# small (or a tmpfs) on some systems.
ARG SLANG_VERSION=v2026.18.2
# libx11-dev: slang-rhi's vulkan backend includes vulkan.h, which pulls in
# the X11 surface headers.
RUN apt-get update && apt-get install -y --no-install-recommends libx11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 --shallow-submodules --recurse-submodules \
        --branch ${SLANG_VERSION} https://github.com/shader-slang/slang.git /root/slang \
    && cmake -S /root/slang -B /root/slang/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DSLANG_ENABLE_TESTS=OFF \
        -DSLANG_ENABLE_EXAMPLES=OFF \
    && cmake --build /root/slang/build --config Release -j"$(nproc)" \
    && mkdir -p /opt/slang/bin /opt/slang/lib \
    && cp /root/slang/build/Release/bin/slangc /opt/slang/bin/slangc \
    && cp /root/slang/build/Release/lib/libslang-compiler.so.0.* /opt/slang/lib/ \
    && cp /root/slang/build/Release/lib/libslang-glsl*.so /opt/slang/lib/ \
    && ln -s /opt/slang/bin/slangc /usr/local/bin/slangc \
    && rm -rf /root/slang

# 4. Rust toolchain.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
# sh.rustup.rs is the installer script; plain rustup.rs serves an HTML
# landing page and cannot be piped to a shell.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

WORKDIR /mnt/workspace
CMD ["/bin/bash"]
