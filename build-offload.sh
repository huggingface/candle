#!/usr/bin/env bash
# Build a candle example against the std::offload GPU backend.
#
#   ./build-offload.sh                 # llama
#   EXAMPLE=llama ./build-offload.sh
#   EXAMPLE=llama2-c FEATURES=candle-datasets ./build-offload.sh
#   TC=nightly-2026-10-06 ./build-offload.sh   # another toolchain, own target dirs
#
# Four passes. The flag placement in each is not interchangeable:
#
#  1. HostMetadata goes to the crate that *defines* the kernels. The manifest is
#     emitted per-crate during that crate's codegen, so pointing it at a
#     downstream crate produces an empty manifest and the kernels are silently
#     invisible.
#  2. Device goes via `--` rather than RUSTFLAGS, so `-Zbuild-std`'s own crates
#     do not each also emit a device.bin and shadow ours. Both 1 and 2 need
#     `--crate-type lib`, overriding the cdylib in Cargo.toml.
#  3. Host goes to the cdylib, which is a *final linked artifact*. rustc runs the
#     offload lowering during fat LTO, and fat LTO never runs for an rlib, so a
#     library crate fails with "requires -C lto=fat" -- and passing -Clto=fat to
#     an rlib silently emits no artifact at all. This is why the kernels are a
#     cdylib rather than a cargo dependency of candle-core.
#  4. candle itself is then built with no offload flags whatsoever. It only
#     needs the cdylib and libomp/libomptarget on the link path and in the rpath.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
GPU="${GPU:-gfx1201}"
EXAMPLE="${EXAMPLE:-llama}"
TC="${TC:-nightly}"
FEATURES="offload${FEATURES:+,$FEATURES}"
SUF="${TC#nightly}"   # "" for plain nightly, "-2026-10-06" otherwise
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# cargo output goes under ~/.cache: this checkout's path contains spaces and
# -Zoffload splits its flag values on whitespace.
OUT="${OUT:-$HOME/.cache/rust-offload/candle-target$SUF}"
KOUT="$HOME/.cache/rust-offload/candle-kernels-target$SUF"
MANIFEST="$HOME/.cache/rust-offload/candle$SUF.manifest"
LIBDIR="$(rustc +"$TC" --print target-libdir)"
mkdir -p "$OUT" "$KOUT"

echo "==> pass 1/4  HostMetadata"
( cd "$PROJ/candle-offload-kernels" && CARGO_TARGET_DIR="$KOUT" \
  cargo +"$TC" rustc -r --lib --crate-type lib -- \
      -Zoffload=HostMetadata="$MANIFEST" -Zunstable-options )
KERNELS="$(tr -d '\0' < "$MANIFEST" | grep -o '::[a-z0-9_]*' | sort -u | tr -d ':' | tr '\n' ' ')"
[ -n "$KERNELS" ] || { echo "error: empty offload manifest" >&2; exit 1; }
echo "    kernels: $KERNELS"

echo "==> pass 2/4  Device ($GPU)"
( cd "$PROJ/candle-offload-kernels" && CARGO_TARGET_DIR="$KOUT" RUSTFLAGS="-Ctarget-cpu=$GPU" \
  cargo +"$TC" rustc -r --lib --crate-type lib \
      --target amdgcn-amd-amdhsa -Zbuild-std=core -- \
      -Zoffload=Device="$MANIFEST" -Zunstable-options )
DEVICE_BIN="$(find "$KOUT/amdgcn-amd-amdhsa" -name device.bin -path '*candle-offload-kernels*' | head -1)"
[ -n "$DEVICE_BIN" ] || { echo "error: device.bin not produced" >&2; exit 1; }
echo "    device image: $(du -h "$DEVICE_BIN" | cut -f1)"

echo "==> pass 3/4  Host (cdylib)"
( cd "$PROJ/candle-offload-kernels" && CARGO_TARGET_DIR="$KOUT" \
  cargo +"$TC" rustc -r --lib -- \
      -Zoffload=Host="$DEVICE_BIN" -Zunstable-options )
SO="$KOUT/release/libcandle_offload_kernels.so"
[ -f "$SO" ] || { echo "error: $SO not produced" >&2; exit 1; }
echo "    $SO ($(du -h "$SO" | cut -f1))"

echo "==> pass 4/4  candle example: $EXAMPLE (no offload flags)"
cd "$PROJ"
CARGO_TARGET_DIR="$OUT" \
RUSTFLAGS="-L $KOUT/release -L $LIBDIR -Clink-args=-Wl,-rpath,$KOUT/release -Clink-args=-Wl,-rpath,$LIBDIR" \
    cargo +"$TC" build -r -p candle-examples --features "$FEATURES" --example "$EXAMPLE"

echo
echo "built: $OUT/release/examples/$EXAMPLE"
