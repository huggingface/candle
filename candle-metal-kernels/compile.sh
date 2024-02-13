xcrun metal -c  src/gemm/kernels/steel_gemm.metal -I src/
xcrun metallib steel_gemm.air -o src/gemm/steel_gemm.metallib
