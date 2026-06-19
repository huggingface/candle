#include "mmq_common.cuh"
#include "mmq_gguf.cuh"

// Force instantiation of all mmq_x variants for GGML_TYPE_Q8_0
template <int mmq_x>
static void instantiate_mmq_q8_0(float * tmp_fixup,
    const mmq_args & args, cudaStream_t stream,
    int cc, int nsm, size_t smpbo, int warp_size_host) {

    const int mmq_y = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ? 128 : 64;
    const int nwarps = 256 / warp_size_host;
    const int nbytes_shared = mmq_get_nbytes_shared<GGML_TYPE_Q8_0>(mmq_x, mmq_y, cc, warp_size_host, nwarps);
    const int nty = (args.nrows_x + mmq_y - 1) / mmq_y;
    const int ntx = (args.ncols_max + mmq_x - 1) / mmq_x;
    const int ntzw = args.nchannels_y * args.nsamples_y;
    const dim3 block_dims(warp_size_host, nwarps, 1);
    const int channel_ratio = args.nchannels_y / args.nchannels_x;
    const int sample_ratio  = args.nsamples_y  / args.nsamples_x;

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<GGML_TYPE_Q8_0, mmq_x, false>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<GGML_TYPE_Q8_0, mmq_x,  true>), nbytes_shared);

    if (!args.use_stream_k) {
        const dim3 grid(nty, ntx, ntzw);
        if (args.nrows_x % mmq_y == 0) {
            mul_mat_q<GGML_TYPE_Q8_0, mmq_x, false><<<grid, block_dims, nbytes_shared, stream>>>(
                args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                args.ncols_max);
        } else {
            mul_mat_q<GGML_TYPE_Q8_0, mmq_x, true><<<grid, block_dims, nbytes_shared, stream>>>(
                args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr,
                args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
                channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
                sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
                args.ncols_max);
        }
        return;
    }

    // Stream-k
    const dim3 grid_sk(nsm, 1, 1);
    const bool fixup_needed = ntx * nty * ntzw % nsm != 0;
    
    
    if (args.nrows_x % mmq_y == 0) {
        mul_mat_q<GGML_TYPE_Q8_0, mmq_x, false><<<grid_sk, block_dims, nbytes_shared, stream>>>(
            args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup,
            args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
            channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
            sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
            args.ncols_max);
        if (fixup_needed) {
            mul_mat_q_stream_k_fixup<GGML_TYPE_Q8_0, mmq_x, false><<<grid_sk, block_dims, 0, stream>>>(
                args.ids_dst, args.expert_bounds, args.dst, tmp_fixup, args.ncols_x, args.nrows_x, args.ncols_dst,
                args.nrows_dst, args.nchannels_y, args.stride_channel_dst, args.nsamples_y, args.stride_sample_dst,
                args.ncols_max);
        }
    } else {
        mul_mat_q<GGML_TYPE_Q8_0, mmq_x, true><<<grid_sk, block_dims, nbytes_shared, stream>>>(
            args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tmp_fixup,
            args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
            channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
            sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
            args.ncols_max);
        if (fixup_needed) {
            mul_mat_q_stream_k_fixup<GGML_TYPE_Q8_0, mmq_x, true><<<grid_sk, block_dims, 0, stream>>>(
                args.ids_dst, args.expert_bounds, args.dst, tmp_fixup, args.ncols_x, args.nrows_x, args.ncols_dst,
                args.nrows_dst, args.nchannels_y, args.stride_channel_dst, args.nsamples_y, args.stride_sample_dst,
                args.ncols_max);
        }
    }
}

static void launch_mmq_case_q8_0(float * tmp_fixup, const mmq_args & args, cudaStream_t stream,
    int cc, int nsm, size_t smpbo, int warp_size_host) {

    const int mmq_x_max = (turing_mma_available(cc)) ? 128 :
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ? 64 : 64;
    const int mmq_y = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ? 128 : 64;
    const int nwarps = 256 / warp_size_host;

    int mmq_x_best = 0;
    int ntiles_x_best = INT_MAX;
    for (int mmq_x = 8; mmq_x <= mmq_x_max && ntiles_x_best > 1; mmq_x += 8) {
        const int granularity = (turing_mma_available(cc) && mmq_x >= 48) ? 16 : 8;
        if (mmq_x % granularity != 0) continue;
        const size_t nbs = mmq_get_nbytes_shared<GGML_TYPE_Q8_0>(mmq_x, mmq_y, cc, warp_size_host, nwarps);
        if (nbs > smpbo) continue;
        const int ntiles_x = (args.ncols_max + mmq_x - 1) / mmq_x;
        if (ntiles_x < ntiles_x_best) { mmq_x_best = mmq_x; ntiles_x_best = ntiles_x; }
    }

    switch (mmq_x_best) {
        case   8: instantiate_mmq_q8_0<  8>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case  16: instantiate_mmq_q8_0< 16>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case  24: instantiate_mmq_q8_0< 24>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case  32: instantiate_mmq_q8_0< 32>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case  40: instantiate_mmq_q8_0< 40>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case  48: instantiate_mmq_q8_0< 48>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case  56: instantiate_mmq_q8_0< 56>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case  64: instantiate_mmq_q8_0< 64>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case  72: instantiate_mmq_q8_0< 72>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case  80: instantiate_mmq_q8_0< 80>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case  88: instantiate_mmq_q8_0< 88>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case  96: instantiate_mmq_q8_0< 96>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case 104: instantiate_mmq_q8_0<104>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case 112: instantiate_mmq_q8_0<112>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case 120: instantiate_mmq_q8_0<120>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        case 128: instantiate_mmq_q8_0<128>(tmp_fixup, args, stream, cc, nsm, smpbo, warp_size_host); break;
        default: break;
    }
}

extern "C" void launch_mmq_gguf_q8_0(
    void *tmp_fixup_ptr,
    const void *x, const void *y_q8_1_mmq, void *dst,
    int64_t ncols_x, int64_t nrows_x, int64_t ncols_y,
    int64_t stride_row_x, int64_t stride_col_dst,
    int cc, int nsm, int64_t smpbo, int warp_size_host,
    void *stream) {

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA);

    const mmq_args args = {
        (const char *)x, GGML_TYPE_Q8_0, (const int *)y_q8_1_mmq, nullptr, nullptr, (float *)dst,
        ncols_x, nrows_x, ncols_y, stride_row_x, ncols_y, nrows_x,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, ncols_y
    };

    launch_mmq_case_q8_0((float *)tmp_fixup_ptr, args, (cudaStream_t)stream, cc, nsm, smpbo, warp_size_host);
}
