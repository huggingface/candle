
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <cassert>
#include "search_sorted_kernel.cu"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

template <typename input_t, typename output_t>
extern __global__ void searchsorted_cuda_kernel(
    output_t *data_out,
    const input_t *data_in,
    const input_t *data_bd,
    const u_int32_t idim_in,
    const u_int32_t idim_bd,
    const u_int32_t numel_in,
    const bool right,
    const bool is_1d_boundaries,
    const bool is_1d_values);

inline int getMaxThreadsPerBlock()
{
    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    return props.maxThreadsPerBlock;
}

template <typename input_t, typename output_t>
std::vector<output_t> search_sorted(std::vector<input_t> sorted_seq, int innerdim_ss, std::vector<input_t> values, int innerdim_vs, bool right)
{

    assert(values.size() % innerdim_vs == 0);
    assert(sorted_seq.size() % innerdim_ss == 0);

    int num_val_rows = values.size() / innerdim_vs;
    int num_seq_rows = sorted_seq.size() / innerdim_ss;

    bool is_1d_boundaries = num_seq_rows == 1 ? 1 : 0;
    bool is_1d_values = num_val_rows == 1 ? 1 : 0;
    int num_output_rows = std::max(num_val_rows, num_seq_rows);

    std::vector<output_t> output(num_output_rows * innerdim_vs);
    std::fill(output.begin(), output.end(), 0);
    int numel = output.size();
    // Allocate device memory
    input_t *d_tensor;
    input_t *d_values;
    output_t *d_output;
    cudaMalloc(&d_tensor, sorted_seq.size() * sizeof(input_t));
    cudaMalloc(&d_values, values.size() * sizeof(input_t));
    cudaMalloc(&d_output, output.size() * sizeof(output_t));

    // Copy data to device
    cudaMemcpy(d_tensor, sorted_seq.data(), sorted_seq.size() * sizeof(input_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), values.size() * sizeof(input_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int maxThreads = getMaxThreadsPerBlock();
    int maxGrid = 1024;
    dim3 block = dim3(std::min(maxThreads, numel));
    dim3 grid = dim3(std::min(maxGrid, int((numel + block.x - 1) / block.x)));

    // Innermost dimensions: for values, this corresponds to the number of values to search for per row
    searchsorted_cuda_kernel<<<grid, block>>>(d_output, d_values, d_tensor, innerdim_vs, innerdim_ss, numel, right, is_1d_boundaries, is_1d_values);
    cudaDeviceSynchronize();
    // Copy output data back to host
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(output_t), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_tensor);
    cudaFree(d_values);
    cudaFree(d_output);
    return output;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    for (int i = 0; i < v.size(); i++)
    {
        if constexpr (std::is_same<T, __half>::value)
        {
            os << __half2float(v[i]);
        }
        else if constexpr (std::is_same<T, __nv_bfloat16>::value)
        {
            os << __bfloat162float(v[i]);
        }

        else
        {
            os << v[i];
        }
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template <typename input_t, typename output_t>
void run_test(std::string_view test_name, std::vector<input_t> ss, std::vector<input_t> vals, size_t innerdim_ss, size_t innerdim_vs, bool right, std::vector<output_t> expected)
{
    std::vector<output_t> output = search_sorted<input_t, output_t>(ss, innerdim_ss, vals, innerdim_vs, right);
    if (output != expected)
    {
        std::cout << test_name << ": Failed!" << std::endl;
        std::cout << "Test 1-D ss: " << ss << " | 1-D values: " << vals << " Right: " << right << std::endl;
        std::cout << "Expected: " << expected << " -> "
                  << "Got: " << output << std::endl;
    }
    else
    {
        std::cout << test_name << ": Passed!" << std::endl;
        // std::cout << "Test 1-D ss: " << ss << "1-D values: " << vals << "Right: " << right << std::endl;
        // std::cout << "Expected: " << expected << " "
        //           << "Got: " << output << std::endl;
    }
}
int main()
{
    using output_t = int;
    // Test 1-D ss, 1-D values
    run_test<float, output_t>("Float 1-D ss, 1-D vals, left", {1, 3, 5, 7, 9}, {3, 6, 9}, 5, 3, false, {1, 3, 4});
    run_test<float, output_t>("Float 1-D ss, 1-D vals, right", {1, 3, 5, 7, 9}, {3, 6, 9}, 5, 3, true, {2, 3, 5});

    // Test 1-D ss, 2-D values
    run_test<float, output_t>("Float 2-D ss, 1-D vals, left", {1, 3, 5, 7, 9}, {3, 6, 9, 3, 6, 9}, 5, 3, false, {1, 3, 4, 1, 3, 4});
    run_test<float, output_t>("Float 2-D ss, 1-D vals, right", {1, 3, 5, 7, 9}, {3, 6, 9, 3, 6, 9}, 5, 3, true, {2, 3, 5, 2, 3, 5});

    // Test 2-D ss, 1-D values
    run_test<float, output_t>("Float 2-D ss, 1-D vals, left", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9}, 5, 3, false, {1, 3, 4, 1, 2, 4});
    run_test<float, output_t>("Float 2-D ss, 1-D vals, right", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9}, 5, 3, true, {2, 3, 5, 1, 3, 4});

    // Test 2-D ss, 2-D values same
    run_test<float, output_t>("Float 2-D ss, 2-D vals same, left", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 3, 6, 9}, 5, 3, false, {1, 3, 4, 1, 2, 4});
    run_test<float, output_t>("Float 2-D ss, 2-D vals same, right", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 3, 6, 9}, 5, 3, true, {2, 3, 5, 1, 3, 4});

    // Test 2-D ss, 2-D values diff
    run_test<float, output_t>("Float 2-D ss, 2-D vals diff, left", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
    run_test<float, output_t>("Float 2-D ss, 2-D vals diff, right", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

    // F64 Test 2-D ss, 2-D values diff
    run_test<double, output_t>("Float 2-D ss, 2-D vals diff, left", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
    run_test<double, output_t>("Float 2-D ss, 2-D vals diff, right", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

    // Half Test 2-D ss, 2-D values diff
    run_test<half, output_t>("Half 2-D ss, 2-D vals diff, left", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
    run_test<half, output_t>("Half 2-D ss, 2-D vals diff, right", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

    // Half Test 2-D ss, 2-D values diff
    run_test<__half, output_t>("fp16 2-D ss, 2-D vals diff, left", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
    run_test<__half, output_t>("fp16 2-D ss, 2-D vals diff, right", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

    // Half Test 2-D ss, 2-D values diff
    run_test<__nv_bfloat16, output_t>("bf16 2-D ss, 2-D vals diff, left", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
    run_test<__nv_bfloat16, output_t>("bf16 2-D ss, 2-D vals diff, right", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

    // Int32 Test 2-D ss, 2-D values diff
    run_test<u_int8_t, output_t>("U8 2-D ss, 2-D vals diff, left", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
    run_test<u_int8_t, output_t>("U8 2-D ss, 2-D vals diff, right", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

    // Int64 Test 2-D ss, 2-D values diff
    run_test<u_int32_t, output_t>("U32 2-D ss, 2-D vals diff, left", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
    run_test<u_int32_t, output_t>("U32 2-D ss, 2-D vals diff, right", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

    // Int64 Test 2-D ss, 2-D values diff
    run_test<int64_t, output_t>("I64 2-D ss, 2-D vals diff, left", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
    run_test<int64_t, output_t>("I64 2-D ss, 2-D vals diff, right", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

    {
        using output_t = int64_t;
        // Test 2-D ss, 2-D values diff
        run_test<float, output_t>("Float 2-D ss, 2-D vals diff, left, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
        run_test<float, output_t>("Float 2-D ss, 2-D vals diff, right, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

        // F64 Test 2-D ss, 2-D values diff
        run_test<double, output_t>("Float 2-D ss, 2-D vals diff, left, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
        run_test<double, output_t>("Float 2-D ss, 2-D vals diff, right, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

        // Half Test 2-D ss, 2-D values diff
        run_test<half, output_t>("Half 2-D ss, 2-D vals diff, left, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
        run_test<half, output_t>("Half 2-D ss, 2-D vals diff, right, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

        // Half Test 2-D ss, 2-D values diff
        run_test<__half, output_t>("fp16 2-D ss, 2-D vals diff, left, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
        run_test<__half, output_t>("fp16 2-D ss, 2-D vals diff, right, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

        // Half Test 2-D ss, 2-D values diff
        run_test<__nv_bfloat16, output_t>("bf16 2-D ss, 2-D vals diff, left, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
        run_test<__nv_bfloat16, output_t>("bf16 2-D ss, 2-D vals diff, right, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

        // Int32 Test 2-D ss, 2-D values diff
        run_test<u_int8_t, output_t>("U8 2-D ss, 2-D vals diff, left, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
        run_test<u_int8_t, output_t>("U8 2-D ss, 2-D vals diff, right, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

        // Int64 Test 2-D ss, 2-D values diff
        run_test<u_int32_t, output_t>("U32 2-D ss, 2-D vals diff, left, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
        run_test<u_int32_t, output_t>("U32 2-D ss, 2-D vals diff, right, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});

        // Int64 Test 2-D ss, 2-D values diff
        run_test<int64_t, output_t>("I64 2-D ss, 2-D vals diff, left, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, false, {1, 3, 4, 0, 0, 1});
        run_test<int64_t, output_t>("I64 2-D ss, 2-D vals diff, right, Output i64", {1, 3, 5, 7, 9, 2, 4, 6, 8, 10}, {3, 6, 9, 1, 2, 3}, 5, 3, true, {2, 3, 5, 0, 1, 1});
    }
}
