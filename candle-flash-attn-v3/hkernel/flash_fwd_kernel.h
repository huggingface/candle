/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "flash.h"
#include "utils.h"
#include "softmax.h"
#include "tile_scheduler.hpp"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd_sm90_tma.hpp"

namespace flash {

using namespace cute;

template <typename Ktraits, bool Is_causal, bool Is_local, typename TileScheduler, typename Seqlen_traits, typename Seqlen_traits_Q = Seqlen_traits>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal, Is_local, Seqlen_traits, Seqlen_traits_Q>::Params const mainloop_params,
                    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits, Seqlen_traits_Q>::Params const epilogue_params,
                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params,
                    Seqlen_traits_Q seqlen_traits_q, Seqlen_traits seqlen_traits_k
                    ) {

    using Element = typename Ktraits::Element;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static_assert(Ktraits::Is_WS);
    static constexpr bool Is_WS = Ktraits::Is_WS;
    static constexpr bool No_smem_O = Ktraits::No_smem_O;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
    static constexpr int NumCopyThreads = !Is_WS ? 0 : cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockH = Ktraits::kBlockH;
    // static constexpr int kBlockN = Ktraits::kBlockN;
    // static constexpr int kHeadDim = Ktraits::kHeadDim;

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Is_causal, Is_local, Seqlen_traits, Seqlen_traits_Q>;
    using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits, Seqlen_traits_Q>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // Obtain warp index
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    pipeline_params.role = warp_group_idx == 0
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_Q.init(1 /*numThreads*/);
        if constexpr (!No_smem_O) { shared_storage.barrier_O.init(size(ClusterShape{}) /*numThreads*/); }
    }
    // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params, ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params, ClusterShape{});

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    // static_assert(Ktraits::kNWarps == 12 || Ktraits::kNWarps == 16);
    static_assert(Ktraits::kNWarps == 8 || Ktraits::kNWarps == 12 || Ktraits::kNWarps == 16);
    if (warp_group_idx == 0) {  // Producer
        cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 12 ? 24 : 32>();

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if (warp_idx_in_warpgroup == 0) {  // Load Q, K, V
            PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
            PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

            int work_idx = 0;

            TileScheduler scheduler(&shared_storage.tile_count_semaphore);
            for (auto work_tile_info = scheduler.get_initial_work();
                 work_tile_info.is_valid(scheduler_params);
                 work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(scheduler_params, work_tile_info)) {
                auto block_coord = work_tile_info.get_block_coord(scheduler_params);
                auto [m_block, n_split_idx, bidh, bidb] = block_coord;

                seqlen_traits_q.init(bidb);
                seqlen_traits_k.init(bidb);
                if constexpr(seqlen_traits_q.UseVarSeqLen) {
                    // NOTE: to support in future with gqa packed layouts, changed kBlockM to kBlockM/kBlockH
                    if (m_block * (kBlockM/kBlockH) >= seqlen_traits_q.actual_seq_len) {
                        continue;
                    }
                }
                int n_block_min = 0, n_block_max;
                collective_mainloop.get_n_block_min_max(
                        mainloop_params, m_block, n_split_idx, seqlen_traits_q, seqlen_traits_k,
                        n_block_min, n_block_max);
                if constexpr (Is_causal || Is_local || seqlen_traits_k.UseVarSeqLen || Ktraits::Is_split) {
                    if(n_block_max <= n_block_min) {
                        scheduler.prefetch_next_work(scheduler_params, work_tile_info);
                        scheduler.broadcast_next_work(work_tile_info);
                        continue;
                    }
                }
                collective_mainloop.load(
                    mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
                    shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx,
                    seqlen_traits_q, seqlen_traits_k, n_block_min, n_block_max);
                ++work_idx;
            }
            collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
        }
    } else {  // Consumer
        cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 16 ? 160 : Ktraits::kNWarps == 12 ? 240 : 256>();

        TileScheduler scheduler(&shared_storage.tile_count_semaphore);
        // Initialize matmul objects.
        typename Ktraits::TiledMma1 tiled_mma1;

        PipelineState smem_pipe_read_k, smem_pipe_read_v;
        // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
        // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

        collective_mainloop.mma_init();
        scheduler.init_consumer();

        int work_idx = 0;
        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params, work_tile_info)) {
            // Attention output (GEMM-II) accumulator.
            Tensor tOrO = partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
            flash::Softmax<2 * (2 * kBlockM / NumMmaThreads)> softmax(mainloop_params.softmax_scale_log2);

            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, n_split_idx, bidh, bidb] = block_coord;

            seqlen_traits_q.init(bidb);
            seqlen_traits_k.init(bidb);
            if constexpr(seqlen_traits_q.UseVarSeqLen) {
                // NOTE: to support in future with gqa packed layouts, changed kBlockM to kBlockM/kBlockH
                if (m_block * (kBlockM/kBlockH) >= seqlen_traits_q.actual_seq_len) {
                    continue;
                }
            }
            int n_block_max, n_block_min = 0;
            collective_mainloop.get_n_block_min_max(
                    mainloop_params, m_block, n_split_idx, seqlen_traits_q, seqlen_traits_k,
                    n_block_min, n_block_max);
            if constexpr (Is_causal || Is_local || seqlen_traits_k.UseVarSeqLen || Ktraits::Is_split) {
                if(n_block_max <= n_block_min) {  // We exit early and write 0 to gO and -inf to gLSE.
                    if constexpr(!Seqlen_traits_Q::UseGQAPacking) {
                        collective_epilogue.store_zero(epilogue_params, shared_storage, threadIdx.x - NumCopyThreads,
                            block_coord, seqlen_traits_q);
                    } else {
                        collective_epilogue.store_zero_gqa(epilogue_params, shared_storage, threadIdx.x - NumCopyThreads,
                            block_coord, seqlen_traits_q, mainloop_params.qhead_per_khead_divmod);
                    }
                    continue;
                }   
            }         

            collective_mainloop.mma(
                mainloop_params, pipeline_k, pipeline_v, smem_pipe_read_k, smem_pipe_read_v,
                tOrO, softmax, n_block_min, n_block_max, threadIdx.x - NumCopyThreads, work_idx,
                m_block, shared_storage, seqlen_traits_q, seqlen_traits_k);
                // tOrO, softmax, n_block_max, threadIdx.x - NumCopyThreads + (work_idx >> 30), work_idx, shared_storage);
            collective_epilogue.store(
                epilogue_params, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                threadIdx.x - NumCopyThreads, block_coord, seqlen_traits_q, mainloop_params.qhead_per_khead_divmod);

            ++work_idx;
        }
        collective_epilogue.store_tail();
    }

}

template <typename Ktraits, bool Is_causal, bool Is_local, typename TileScheduler, typename Seqlen_traits, typename Seqlen_traits_Q = Seqlen_traits>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws_fp8(CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal, Is_local, Seqlen_traits, Seqlen_traits_Q>::Params const mainloop_params,
                        CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits, Seqlen_traits_Q>::Params const epilogue_params,
                        CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params,
                        Seqlen_traits_Q seqlen_traits_q, Seqlen_traits seqlen_traits_k
                        ) {

    using Element = typename Ktraits::Element;
    static_assert(cutlass::sizeof_bits_v<Element> == 8);
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static_assert(Ktraits::Is_WS);
    static constexpr bool Is_WS = Ktraits::Is_WS;
    static constexpr bool No_smem_O = Ktraits::No_smem_O;

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
    static constexpr int NumCopyThreads = !Is_WS ? 0 : cutlass::NumThreadsPerWarpGroup;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockH = Ktraits::kBlockH;
    // static constexpr int kBlockN = Ktraits::kBlockN;
    // static constexpr int kHeadDim = Ktraits::kHeadDim;
    static constexpr bool Delay_V_release = Is_causal && Ktraits::kHeadDim == 128 && Ktraits::kNWarps != 8;    
    static constexpr bool Use_max_offset = true;

    using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Is_causal, Is_local, Seqlen_traits, Seqlen_traits_Q>;
    using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits, Seqlen_traits_Q>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using MainloopPipelineVt = typename Ktraits::MainloopPipelineNoTMA;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineParamsVt = typename MainloopPipelineVt::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

    int const lane_predicate = cute::elect_one_sync();
    int const warp_idx = cutlass::canonical_warp_idx_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
        CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
    }

    // Obtain warp index
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    // additional pipeline to synchronize out-of-place smem transpose of V
    PipelineParamsVt pipeline_params_vt;
    pipeline_params_vt.producer_arv_count = NumCopyThreads;
    pipeline_params_vt.consumer_arv_count = NumMmaThreads;
    MainloopPipelineVt pipeline_vt(shared_storage.pipeline_vt, pipeline_params_vt);
    
    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    pipeline_params.role = warp_group_idx == 0
        ? MainloopPipeline::ThreadCategory::Producer
        : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NumMmaThreads;

    if (warp_idx == 0 && lane_predicate) {
        shared_storage.barrier_Q.init(1 /*numThreads*/);
        if constexpr (!No_smem_O) { shared_storage.barrier_O.init(size(ClusterShape{}) /*numThreads*/); }
    }
    // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params, ClusterShape{});
    // pipeline_v has producer warpgroup for its consumer in fp8 kernel
    pipeline_params.num_consumers = NumCopyThreads;
    pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params, ClusterShape{});

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    float descale_q = *mainloop_params.descale_q_ptr;
    float descale_k = *mainloop_params.descale_k_ptr;
    float descale_v = *mainloop_params.descale_v_ptr;
    shared_storage.softmax_scale_qk_log2 = mainloop_params.softmax_scale_log2 * descale_q * descale_k;
    shared_storage.descale_v = descale_v;
    shared_storage.seqlen_init_k = seqlen_traits_k.UseVarSeqLen || bool(seqlen_traits_k.seq_used);

    // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    static_assert(Ktraits::kNWarps == 8 || Ktraits::kNWarps == 12 || Ktraits::kNWarps == 16);
    if (warp_group_idx == 0) {  // Producer
        cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 16 ? 32 : Ktraits::kNWarps == 12 ? 40 : 56>();
            
        PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>(); 
        PipelineState smem_pipe_read, smem_pipe_release;

        int work_idx = 0;

        TileScheduler scheduler(&shared_storage.tile_count_semaphore);
        for (auto work_tile_info = scheduler.get_initial_work();
                work_tile_info.is_valid(scheduler_params);
                work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(scheduler_params, work_tile_info)) {
            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, n_split_idx, bidh, bidb] = block_coord;

            if constexpr (seqlen_traits_q.UseVarSeqLen) { seqlen_traits_q.init(bidb); }
            if (shared_storage.seqlen_init_k) { seqlen_traits_k.init_no_guard(bidb); }
            if constexpr(seqlen_traits_q.UseVarSeqLen) {
                // NOTE: to support in future with gqa packed layout, changed kBlockM to kBlockM/kBlockH
                if (m_block * (kBlockM/kBlockH) >= seqlen_traits_q.actual_seq_len) {
                    continue;
                }
            }
            int n_block_min = 0, n_block_max;
            collective_mainloop.get_n_block_min_max(
                    mainloop_params, m_block, n_split_idx, seqlen_traits_q, seqlen_traits_k,
                    n_block_min, n_block_max);
            if constexpr (Is_causal || Is_local ||seqlen_traits_k.UseVarSeqLen || Ktraits::Is_split) {
                if(n_block_max <= n_block_min) {
                    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
                    scheduler.broadcast_next_work(work_tile_info);
                    // need to sync producer warpgroup
                    cutlass::arch::NamedBarrier::sync(NumCopyThreads, static_cast<int>(FwdNamedBarriers::ProducerWG) /*id*/);
                    continue;
                }
            }
            collective_mainloop.load_fp8(
                mainloop_params, pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, smem_pipe_read,
                shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx,
                seqlen_traits_q, seqlen_traits_k, n_block_min, n_block_max);
            ++work_idx;
            // don't need to sync producer warpgroup here
            // if constexpr (Is_causal) {
            //     cutlass::arch::NamedBarrier::sync(NumCopyThreads, static_cast<int>(FwdNamedBarriers::ProducerWG) /*id*/); }
        }
        collective_mainloop.load_tail_one_write(pipeline_k, pipeline_v, smem_pipe_write);
    } else {  // Consumer
        cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 16 ? 160 : Ktraits::kNWarps == 12 ? 232 : 256>();        

        TileScheduler scheduler(&shared_storage.tile_count_semaphore);
        // Initialize matmul objects.
        typename Ktraits::TiledMma1 tiled_mma1;
        PipelineState smem_pipe_read;
        PipelineState smem_pipe_release;

        collective_mainloop.mma_init();
        scheduler.init_consumer();

        int work_idx = 0;

        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work();
             work_tile_info.is_valid(scheduler_params);
             work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params, work_tile_info)) {
            // Attention output (GEMM-II) accumulator.
            Tensor tOrO = partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
            flash::Softmax<2 * (2 * kBlockM / NumMmaThreads), Use_max_offset> softmax(shared_storage.softmax_scale_qk_log2);

            auto block_coord = work_tile_info.get_block_coord(scheduler_params);
            auto [m_block, n_split_idx, bidh, bidb] = block_coord;

            if constexpr (seqlen_traits_q.UseVarSeqLen) { seqlen_traits_q.init(bidb); }
            if (shared_storage.seqlen_init_k) { seqlen_traits_k.init_no_guard(bidb); }
            if constexpr(seqlen_traits_q.UseVarSeqLen) {
                // NOTE: to support in future with gqa packed layout, changed kBlockM to kBlockM/kBlockH
                if (m_block * (kBlockM/kBlockH) >= seqlen_traits_q.actual_seq_len) {
                    continue;
                }
            }
            int n_block_max, n_block_min = 0;
            collective_mainloop.get_n_block_min_max(
                    mainloop_params, m_block, n_split_idx, seqlen_traits_q, seqlen_traits_k,
                    n_block_min, n_block_max);
            if constexpr (Is_causal || Is_local || seqlen_traits_k.UseVarSeqLen || Ktraits::Is_split) {
                if(n_block_max <= n_block_min) {  // We exit early and write 0 to gO and -inf to gLSE.
                    if constexpr(!Seqlen_traits_Q::UseGQAPacking) {
                        collective_epilogue.store_zero(epilogue_params, shared_storage, threadIdx.x - NumCopyThreads,
                            block_coord, seqlen_traits_q);
                    } else {
                        collective_epilogue.store_zero_gqa(epilogue_params, shared_storage, threadIdx.x - NumCopyThreads,
                            block_coord, seqlen_traits_q, mainloop_params.qhead_per_khead_divmod);
                    }
                    continue;
                }
            }
            
            collective_mainloop.mma_fp8<Delay_V_release>(
                mainloop_params, pipeline_k, pipeline_vt, smem_pipe_read, smem_pipe_release,
                tOrO, softmax, n_block_min, n_block_max, threadIdx.x - NumCopyThreads, work_idx, m_block,
                shared_storage, seqlen_traits_q, seqlen_traits_k);

            collective_epilogue.store(
                epilogue_params, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                threadIdx.x - NumCopyThreads, block_coord, seqlen_traits_q, mainloop_params.qhead_per_khead_divmod);

            ++work_idx;
        }
        collective_epilogue.store_tail();
    }

}

} // namespace flash
