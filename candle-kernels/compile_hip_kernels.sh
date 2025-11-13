#!/bin/bash
# TEAM-491: Compile HIP kernels to HSACO binaries
# Compiles all .hip files in src/hip/ to .hsaco binaries

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src/hip"
OUT_DIR="${SCRIPT_DIR}/hsaco"

# Create output directory
mkdir -p "${OUT_DIR}"

# Check if hipcc is available
if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found. Please install ROCm."
    exit 1
fi

echo "Compiling HIP kernels to HSACO..."
echo "Source: ${SRC_DIR}"
echo "Output: ${OUT_DIR}"
echo ""

# Compile each .hip file
for hip_file in "${SRC_DIR}"/*.hip; do
    if [ -f "$hip_file" ]; then
        filename=$(basename "$hip_file" .hip)
        hsaco_file="${OUT_DIR}/${filename}.hsaco"
        
        echo "Compiling ${filename}.hip -> ${filename}.hsaco"
        
        hipcc --genco \
            --offload-arch=gfx90a \
            --offload-arch=gfx940 \
            --offload-arch=gfx1030 \
            --offload-arch=gfx1100 \
            -I"${SRC_DIR}" \
            -O3 \
            -ffast-math \
            -o "${hsaco_file}" \
            "${hip_file}"
        
        echo "  ✓ Created ${hsaco_file}"
    fi
done

echo ""
echo "Compilation complete!"
echo "HSACO files:"
ls -lh "${OUT_DIR}"/*.hsaco
