#!/bin/bash
# Created by: TEAM-488 (Phase 2)
# Compile HIP kernels to HSACO binaries
#
# This script compiles all .hip files to .hsaco binaries using hipcc.
# The .hsaco files are AMD's equivalent of CUDA's .cubin files.
#
# Requirements:
# - ROCm installed (provides hipcc)
# - HIP kernels in src/hip/
#
# Usage:
#   ./compile_kernels.sh [--arch gfx90a]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=== HIP Kernel Compilation Script ==="
echo ""

# Check if hipcc is available
if ! command -v hipcc &> /dev/null; then
    echo -e "${RED}ERROR: hipcc not found!${NC}"
    echo ""
    echo "hipcc is part of ROCm. Install ROCm to get it:"
    echo "  Ubuntu: sudo apt install rocm-hip-sdk"
    echo "  Or download from: https://rocm.docs.amd.com/"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ hipcc found: $(hipcc --version | head -n1)${NC}"
echo ""

# Parse arguments
TARGET_ARCH="${HIP_ARCH:-gfx90a}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            TARGET_ARCH="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Target architecture: $TARGET_ARCH${NC}"
echo ""

# Directories
HIP_DIR="src/hip"
HSACO_DIR="hsaco"

# Create output directory
mkdir -p "$HSACO_DIR"

# Check if HIP directory exists
if [ ! -d "$HIP_DIR" ]; then
    echo -e "${RED}ERROR: $HIP_DIR not found!${NC}"
    echo "Run ./translate_to_hip.sh first to generate HIP kernels."
    exit 1
fi

# Count HIP files
hip_count=$(find "$HIP_DIR" -name "*.hip" | wc -l)
if [ "$hip_count" -eq 0 ]; then
    echo -e "${RED}ERROR: No .hip files found in $HIP_DIR${NC}"
    echo "Run ./translate_to_hip.sh first to generate HIP kernels."
    exit 1
fi

echo "Found $hip_count HIP kernel files"
echo ""
echo "=== Compiling Kernels ==="
echo ""

# Compilation flags
HIPCC_FLAGS=(
    "--genco"                    # Generate code object
    "--offload-arch=$TARGET_ARCH" # Target GPU architecture
    "-O3"                        # Optimization level
    "-I$HIP_DIR"                 # Include HIP headers
    "-ffast-math"                # Fast math
    "-fgpu-rdc"                  # Relocatable device code
)

# Compile each kernel
success_count=0
fail_count=0

for hip_file in "$HIP_DIR"/*.hip; do
    if [ ! -f "$hip_file" ]; then
        continue
    fi
    
    name=$(basename "$hip_file" .hip)
    output="$HSACO_DIR/${name}.hsaco"
    
    echo -n "Compiling: $name ... "
    
    if hipcc "${HIPCC_FLAGS[@]}" "$hip_file" -o "$output" 2>&1 | tee /tmp/hipcc_${name}.log | grep -q "error:"; then
        echo -e "${RED}✗ FAILED${NC}"
        echo "  See /tmp/hipcc_${name}.log for details"
        ((fail_count++))
    else
        if [ -f "$output" ]; then
            size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output")
            echo -e "${GREEN}✓ OK${NC} ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "$size bytes"))"
            ((success_count++))
        else
            echo -e "${RED}✗ FAILED${NC} (output not created)"
            ((fail_count++))
        fi
    fi
done

echo ""
echo "=== Compilation Summary ==="
echo ""
echo -e "Success: ${GREEN}$success_count${NC}"
echo -e "Failed:  ${RED}$fail_count${NC}"
echo ""

if [ "$fail_count" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Some kernels failed to compile${NC}"
    echo "Check /tmp/hipcc_*.log for error details"
    exit 1
fi

echo "Compiled binaries:"
ls -lh "$HSACO_DIR" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo -e "${GREEN}All kernels compiled successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Update build.rs to embed these .hsaco files"
echo "  2. Create KernelCache for runtime loading"
echo "  3. Test kernel execution"
