#!/bin/sh
# Build the candle Vulkan development container image.
#
# Usage: .devops/build-vulkan-dev.sh
#
# The image is tagged candle-vulkan-dev (override with CANDLE_VULKAN_IMAGE).
# The Dockerfile lives in this directory and copies nothing from the
# repository, so the build context is this directory only; the checkout
# (including target/) is not sent to the container engine.
set -eu

# shellcheck disable=SC1007 # CDPATH= is a temporary empty assignment prefix
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IMAGE=${CANDLE_VULKAN_IMAGE:-candle-vulkan-dev}

docker build -f "$SCRIPT_DIR/vulkan.Dockerfile" -t "$IMAGE" "$SCRIPT_DIR"
