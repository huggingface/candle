#!/bin/sh
# Run the candle Vulkan development container.
#
# Usage:
#   .devops/run-vulkan-dev.sh                     # interactive shell
#   .devops/run-vulkan-dev.sh -- cargo test ...   # run a command instead
#
# The host checkout ($HOME/workspace/candle, override with CANDLE_DIR) is
# mounted at /mnt/workspace, /dev/dri is passed through for GPU access, and
# the host's Hugging Face cache is mounted so models (and the HF token) do
# not have to be re-downloaded inside the container.
#
# Environment variables set on the host are forwarded into the container:
#   CANDLE_BACKEND          example backend selection (cpu, cuda, metal, vulkan)
#   CANDLE_VULKAN_TEST_GPU  vulkan device index for the test suite (0/1)
#   CANDLE_VULKAN_SLANG_OPT slangc optimization override for shader builds
#   MESA_VK_DEVICE_SELECT   pin the radv device by PCI id
#   HF_TOKEN                Hugging Face access token
set -eu

IMAGE=${CANDLE_VULKAN_IMAGE:-candle-vulkan-dev}
CANDLE_DIR=${CANDLE_DIR:-$HOME/workspace/candle}

if [ ! -d "$CANDLE_DIR" ]; then
    echo "error: candle checkout not found at $CANDLE_DIR (set CANDLE_DIR)" >&2
    exit 1
fi

# Allow a leading "--" to separate script options from the container
# command.
if [ "${1:-}" = "--" ]; then
    shift
fi

# MESA_VK_DEVICE_SELECT pins the device ordering (e.g. the R9700 first on
# the arch machine). When the caller is not a login shell the variable may
# be unset even though /etc/environment defines it; fall back to that value
# so the container's device indices match the host's.
if [ -z "${MESA_VK_DEVICE_SELECT:-}" ] && [ -r /etc/environment ]; then
    MESA_VK_DEVICE_SELECT=$(sed -n 's/^MESA_VK_DEVICE_SELECT=//p' /etc/environment | tail -n 1)
fi

# Forward selected host environment variables when set.
ENV_ARGS=""
for name in CANDLE_BACKEND CANDLE_VULKAN_TEST_GPU CANDLE_VULKAN_SLANG_OPT MESA_VK_DEVICE_SELECT HF_TOKEN; do
    # shellcheck disable=SC2087 # fixed list of variable names
    eval "value=\${$name:-}"
    if [ -n "$value" ]; then
        ENV_ARGS="${ENV_ARGS} -e ${name}=${value}"
    fi
done

# Mount the host's Hugging Face cache when present.
HF_MOUNT=""
HF_CACHE=${HOME}/.cache/huggingface
if [ -d "$HF_CACHE" ]; then
    HF_MOUNT="--volume ${HF_CACHE}:/root/.cache/huggingface"
fi

# Allocate a TTY only when attached to one, so the script also works
# piped or from automation.
TTY_FLAG="-i"
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAG="-it"
fi

# shellcheck disable=SC2086
docker run ${TTY_FLAG} --rm \
    --device /dev/dri \
    --volume "${CANDLE_DIR}:/mnt/workspace" \
    ${HF_MOUNT} \
    ${ENV_ARGS} \
    "$IMAGE" "$@"
