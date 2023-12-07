
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <cassert>
#include "search_sorted_kernel.cu"

template <typename input_t, typename output_t>
extern __global__ void searchsorted_cuda_kernel(
    output_t *data_out,
    const input_t *data_in,
    const input_t *data_bd,
    int idim_in,
    int idim_bd,
    int numel_in,
    bool right,
    bool is_1d_boundaries,
    bool is_1d_values);

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
    int *d_output;
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
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template <typename T>
void run_test(std::vector<T> ss, std::vector<T> vals, size_t innerdim_ss, size_t innerdim_vs, std::vector<int> expected)
{
    std::vector<int> output = search_sorted<T, int>(ss, innerdim_ss, vals, innerdim_vs, false);
    if (output != expected)
    {
        std::cout << "Test failed" << std::endl;
        std::cout << "Test 1-D ss: " << ss << "1-D values: " << vals << std::endl;
        std::cout << "Expected: " << expected << " "
                  << "Got: " << output << std::endl;
    }
    else
    {
        std::cout << "Test passed!" << std::endl;
    }
}
int main()
{
    // Test 1-D ss, 1-D values
    run_test<float>({1, 3, 5, 7, 9}, {3, 6, 9}, 5, 3, {1, 3, 4});
}
