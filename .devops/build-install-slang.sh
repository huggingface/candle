#!/bin/sh
# build-install-slang.sh
#
# Build the shader-slang programs (slangc and friends) from a fresh source
# clone and install them into a directory.
#
# What it does:
#   1. Clones shader-slang/slang at a release tag into a script-managed
#      source directory under $HOME. It never reads, modifies, or deletes any
#      other clone of the shader-slang code (for example a checkout under
#      $HOME/workspace/slang): the source it uses always lives in SRC_DIR.
#   2. Builds the programs with CMake using -DSLANG_LIB_TYPE=STATIC. With the
#      compiler library built statically, the resulting executables are mostly
#      self-contained: the only dynamic dependencies are the usual system
#      libraries (libc, libm, libstdc++, libpthread, ...).
#   3. Installs the built programs into the install directory, plus the
#      downstream-compiler plugin libraries they dlopen at runtime into
#      <install-dir>/../lib (where the binaries' RUNPATH looks).
#   4. Verifies the installed slangc runs and lists its dynamic dependencies.
#
# It is used in two ways:
#   - On the host, to install a recent slangc next to the user's tools
#     (default install directory: $HOME/.local/bin).
#   - Inside the candle Vulkan development container (vulkan.Dockerfile), to
#     install the programs under /usr/local/bin.
#
# Usage:
#   build-install-slang.sh [options]
#
# Options:
#   --install-dir DIR   Install directory (default: $HOME/.local/bin)
#   --version TAG       Release tag to build (default: v2026.20)
#   --src-dir DIR       Source clone location (default: $HOME/.local/share/slang-src)
#   --build-dir DIR     CMake build directory (default: $HOME/.local/share/slang-build)
#   --jobs N            Parallel build jobs (default: nproc)
#   --clean             Remove the source and build directories after a
#                       successful install (used by the container build so the
#                       image only keeps the installed programs)
#   -h, --help          Show this help and exit
#
# Build prerequisites: git, cmake, a C/C++ toolchain. Tests, examples, the
# slang-rhi host interface and gfx are all disabled, so no extra system
# development packages are required beyond the base toolchain.
#
# This is plain POSIX sh (runs under /bin/sh, verified with shellcheck -s dash).

set -eu

SLANG_VERSION_DEFAULT=v2026.20
SLANG_REPO=https://github.com/shader-slang/slang.git
# Programs to install; each one is copied from the build bin directory when
# it exists there. slangc is the one candle's build.rs invokes; the rest are
# installed for convenience.
SLANG_PROGRAMS="slangc slanganalyzer slangd slangi slang"

INSTALL_DIR="${HOME}/.local/bin"
SLANG_VERSION="${SLANG_VERSION_DEFAULT}"
SRC_DIR="${HOME}/.local/share/slang-src"
BUILD_DIR="${HOME}/.local/share/slang-build"
JOBS=""
CLEAN=0

usage() {
    sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --install-dir) INSTALL_DIR="${2:?--install-dir requires a value}"; shift 2 ;;
        --version) SLANG_VERSION="${2:?--version requires a value}"; shift 2 ;;
        --src-dir) SRC_DIR="${2:?--src-dir requires a value}"; shift 2 ;;
        --build-dir) BUILD_DIR="${2:?--build-dir requires a value}"; shift 2 ;;
        --jobs) JOBS="${2:?--jobs requires a value}"; shift 2 ;;
        --clean) CLEAN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for tool in git cmake; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool not found on PATH: $tool" >&2
        exit 1
    fi
done

if [ -z "$JOBS" ]; then
    if command -v nproc >/dev/null 2>&1; then
        JOBS=$(nproc)
    else
        JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
    fi
fi

echo "==> shader-slang build and install"
echo "    version:     $SLANG_VERSION"
echo "    src-dir:     $SRC_DIR"
echo "    build-dir:   $BUILD_DIR"
echo "    install-dir: $INSTALL_DIR"
echo "    jobs:        $JOBS"

# --- 1. source clone (fresh, under $HOME) --------------------------------
# The clone lives in SRC_DIR, a location this script manages. It is a distinct
# path from any user clone, which is never touched. If SRC_DIR already holds a
# clone made by this script, reuse it (fetch the requested tag and sync
# submodules); otherwise perform a fresh shallow clone.
if [ -d "$SRC_DIR/.git" ]; then
    echo "==> reusing script clone at $SRC_DIR (checking out $SLANG_VERSION)"
    git -C "$SRC_DIR" fetch --depth 1 origin "$SLANG_VERSION"
    git -C "$SRC_DIR" checkout -f "$SLANG_VERSION"
    git -C "$SRC_DIR" submodule update --init --recursive --depth 1
else
    if [ -e "$SRC_DIR" ]; then
        echo "error: $SRC_DIR exists but is not a git checkout; refusing to modify it" >&2
        exit 1
    fi
    echo "==> fresh shallow clone of $SLANG_VERSION into $SRC_DIR"
    git clone --depth 1 --branch "$SLANG_VERSION" \
        --recurse-submodules --shallow-submodules \
        "$SLANG_REPO" "$SRC_DIR"
fi

# --- 2. configure and build (static) ------------------------------------
# STATIC library type makes the programs link the compiler statically. Tests,
# examples, slang-rhi and gfx are disabled so the build is lean and produces
# only the standalone programs.
echo "==> configuring (CMAKE_BUILD_TYPE=Release, SLANG_LIB_TYPE=STATIC)"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSLANG_LIB_TYPE=STATIC \
    -DSLANG_ENABLE_TESTS=OFF \
    -DSLANG_ENABLE_EXAMPLES=OFF \
    -DSLANG_ENABLE_SLANG_RHI=OFF \
    -DSLANG_ENABLE_GFX=OFF

echo "==> building (-j$JOBS)"
cmake --build "$BUILD_DIR" --config Release -j "$JOBS"

# --- 3. install ----------------------------------------------------------
# Locate the build output bin directory by finding the slangc executable.
SLANGC_BUILT=$(find "$BUILD_DIR" -type f -name slangc | head -n 1)
if [ -z "$SLANGC_BUILT" ]; then
    echo "error: build did not produce a slangc executable under $BUILD_DIR" >&2
    exit 1
fi
BIN_DIR=$(dirname "$SLANGC_BUILT")
echo "==> build bin directory: $BIN_DIR"

mkdir -p "$INSTALL_DIR"
installed=""
# shellcheck disable=SC2086 # word splitting of SLANG_PROGRAMS is intended
for prog in $SLANG_PROGRAMS; do
    if [ -f "$BIN_DIR/$prog" ]; then
        cp -f "$BIN_DIR/$prog" "$INSTALL_DIR/$prog"
        chmod 0755 "$INSTALL_DIR/$prog"
        installed="${installed} $prog"
    fi
done
echo "==> installed:$installed -> $INSTALL_DIR"

# The programs are linked with SLANG_LIB_TYPE=STATIC, but the downstream
# compiler backends (glslang, for the SPIR-V/GLSL codegen path) are loaded at
# runtime via dlopen. The binaries carry RUNPATH "$ORIGIN/../lib:$ORIGIN:", so
# the plugin shared objects must live in <install-dir>/../lib for the
# binaries to find them. These plugins depend only on the standard system
# libraries (libstdc++, libm, libc, ...).
LIB_SRC=$(dirname "$BIN_DIR")/lib
LIB_DIR=$(dirname "$INSTALL_DIR")/lib
if [ -d "$LIB_SRC" ]; then
    mkdir -p "$LIB_DIR"
    for so in "$LIB_SRC"/libslang-*.so; do
        [ -e "$so" ] || continue
        cp -f "$so" "$LIB_DIR/"
        chmod 0644 "$LIB_DIR/${so##*/}"
    done
    echo "==> installed runtime plugin libraries -> $LIB_DIR"
fi

# --- 4. verify -----------------------------------------------------------
INSTALLED_SLANGC="$INSTALL_DIR/slangc"
if [ ! -x "$INSTALLED_SLANGC" ]; then
    echo "error: $INSTALLED_SLANGC is not an installed executable" >&2
    exit 1
fi
echo "==> $INSTALLED_SLANGC -version"
"$INSTALLED_SLANGC" -version
if command -v ldd >/dev/null 2>&1; then
    echo "==> dynamic dependencies of $INSTALLED_SLANGC"
    ldd "$INSTALLED_SLANGC"
fi

# --- 5. optional clean-up ------------------------------------------------
if [ "$CLEAN" -eq 1 ]; then
    echo "==> removing $SRC_DIR and $BUILD_DIR (--clean)"
    rm -rf "$SRC_DIR" "$BUILD_DIR"
fi

echo "==> done"
