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
#                               from source (see step 3), used by
#                               candle-vulkan-kernels/build.rs to compile
#                               the .slang shaders to SPIR-V
#
# Build with .devops/build-vulkan-dev.sh, run with .devops/run-vulkan-dev.sh.

FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

# 1. Core build tools.
# The -dev packages are used instead of the versioned runtime libraries
# (libncurses-dev pulls in the libtinfo6 runtime, libxml2-dev pulls in
# libxml2-16 on 26.04) so the image does not depend on the exact
# versioned runtime package names.
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
    && rm -rf /var/lib/apt/lists/*

# 2. Vulkan loader, tools, Mesa ICD drivers (AMD, Intel, software) and
# validation layers. (libvulkan-dev pulls in the libvulkan1 loader.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    vulkan-validationlayers \
    && rm -rf /var/lib/apt/lists/*

# 3. Slang shader compiler, built from source by build-install-slang.sh.
# Ubuntu's slang-compiler package (2026.1.1) is too old for the candle
# shaders: the native FP8 kernels use the FloatE4M3 type, which only exists
# in newer Slang versions. The script (shared with the host) clones a pinned
# release under $HOME, builds it with SLANG_LIB_TYPE=STATIC (so the programs
# are mostly self-contained), installs them under /usr/local/bin, and removes
# the source and build trees (--clean) so the image only keeps the binaries.
# Bump SLANG_VERSION to match the version installed on the host.
ARG SLANG_VERSION=v2026.21
# libx11-dev: the SPIR-V/Vulkan codegen path pulls in the X11 surface headers.
COPY build-install-slang.sh /tmp/build-install-slang.sh
RUN apt-get update && apt-get install -y --no-install-recommends libx11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && sh /tmp/build-install-slang.sh --version "${SLANG_VERSION}" --install-dir /usr/local/bin --clean

# 4. Rust toolchain.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
# sh.rustup.rs is the installer script; plain rustup.rs serves an HTML
# landing page and cannot be piped to a shell.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

WORKDIR /mnt/workspace
CMD ["/bin/bash"]
