#!/bin/bash
# Created by: TEAM-488 (Phase 2)
# Translate CUDA kernels to HIP using AMD's hipify-clang tool
# 
# This script uses hipify-clang (AMD's official CUDA→HIP translator)
# to automatically translate all CUDA kernels to HIP.
#
# Requirements:
# - ROCm installed (provides hipify-clang)
# - CUDA headers (for parsing, can be dummy)
#
# Usage:
#   ./translate_to_hip.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== CUDA to HIP Translation Script ==="
echo ""

# Check if hipify-clang is available
if ! command -v hipify-clang &> /dev/null; then
    echo -e "${RED}ERROR: hipify-clang not found!${NC}"
    echo ""
    echo "hipify-clang is part of ROCm. Install ROCm to get it:"
    echo "  Ubuntu: sudo apt install rocm-hip-sdk"
    echo "  Or download from: https://rocm.docs.amd.com/"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ hipify-clang found${NC}"
echo ""

# Directories
SRC_DIR="src"
HIP_DIR="src/hip"
CUDA_PATH="${CUDA_PATH:-/opt/rocm/include}"

# Create output directory
mkdir -p "$HIP_DIR"

# List of CUDA kernels to translate
KERNELS=(
    "affine"
    "binary"
    "cast"
    "conv"
    "fill"
    "indexing"
    "quantized"
    "reduce"
    "sort"
    "ternary"
    "unary"
)

# List of header files to translate
HEADERS=(
    "cuda_utils.cuh"
    "binary_op_macros.cuh"
    "compatibility.cuh"
)

echo "=== Translating Header Files ==="
echo ""

for header in "${HEADERS[@]}"; do
    src_file="$SRC_DIR/$header"
    
    if [ ! -f "$src_file" ]; then
        echo -e "${YELLOW}⚠ Skipping $header (not found)${NC}"
        continue
    fi
    
    # Determine output extension
    if [[ "$header" == *.cuh ]]; then
        out_file="$HIP_DIR/${header%.cuh}.h"
    else
        out_file="$HIP_DIR/$header"
    fi
    
    echo "Translating: $header → $(basename $out_file)"
    
    hipify-clang \
        "$src_file" \
        --cuda-path="$CUDA_PATH" \
        -I "$SRC_DIR" \
        -o "$out_file" \
        2>&1 | grep -v "warning:" || true
    
    if [ $? -eq 0 ] && [ -f "$out_file" ]; then
        size=$(stat -f%z "$out_file" 2>/dev/null || stat -c%s "$out_file")
        echo -e "${GREEN}✓ $(basename $out_file) ($size bytes)${NC}"
    else
        echo -e "${RED}✗ Failed to translate $header${NC}"
    fi
    echo ""
done

echo "=== Translating Kernel Files ==="
echo ""

for kernel in "${KERNELS[@]}"; do
    src_file="$SRC_DIR/${kernel}.cu"
    out_file="$HIP_DIR/${kernel}.hip"
    
    if [ ! -f "$src_file" ]; then
        echo -e "${YELLOW}⚠ Skipping $kernel (not found)${NC}"
        continue
    fi
    
    echo "Translating: ${kernel}.cu → ${kernel}.hip"
    
    hipify-clang \
        "$src_file" \
        --cuda-path="$CUDA_PATH" \
        -I "$SRC_DIR" \
        -o "$out_file" \
        2>&1 | grep -v "warning:" || true
    
    if [ $? -eq 0 ] && [ -f "$out_file" ]; then
        size=$(stat -f%z "$out_file" 2>/dev/null || stat -c%s "$out_file")
        echo -e "${GREEN}✓ ${kernel}.hip ($size bytes)${NC}"
    else
        echo -e "${RED}✗ Failed to translate $kernel${NC}"
    fi
    echo ""
done

echo "=== Translation Summary ==="
echo ""
echo "Translated files:"
ls -lh "$HIP_DIR" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo -e "${GREEN}Translation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review translated files in $HIP_DIR/"
echo "  2. Run ./compile_kernels.sh to compile to .hsaco"
echo "  3. Test kernels with Rust integration"
