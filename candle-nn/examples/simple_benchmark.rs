// candle-nn/examples/simple_benchmark.rs
//! Simple benchmark comparing `Tensor::cat` vs `BlockManager`.
//! Run with: cargo run --release --example simple_benchmark

use candle_nn::kv_cache::BlockManager;
use candle::{DType, Device, Tensor};
use std::time::Instant;

fn main() {
    let num_heads = 8;
    let head_dim = 64;
    let block_size = 16;
    let num_blocks = 512;
    let seq_len = 128;

    // -------------------------------------------------------------------------
    // 1. OLD WAY: Tensor::cat (O(n) copy)
    // -------------------------------------------------------------------------
    let start = Instant::now();
    let mut cache_k = Tensor::zeros((0, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap();
    let mut cache_v = Tensor::zeros((0, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap();

    for _ in 0..seq_len {
        // FIXED: Using zeros explicitly forces F32, no dtype mismatch!
        let new_k = Tensor::zeros((1, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap();
        let new_v = Tensor::zeros((1, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap();
        cache_k = Tensor::cat(&[&cache_k, &new_k], 0).unwrap();
        cache_v = Tensor::cat(&[&cache_v, &new_v], 0).unwrap();
    }
    let cat_time = start.elapsed();

    // -------------------------------------------------------------------------
    // 2. NEW WAY: BlockManager append (O(1) pre-allocated)
    // -------------------------------------------------------------------------
    let start = Instant::now();
    let manager = BlockManager::new(num_blocks, block_size, num_heads, head_dim);
    let seq_id = 0;

    for _ in 0..seq_len {
        let (block_id, offset) = manager.append_slot(seq_id).unwrap();
        std::hint::black_box((block_id, offset));
    }
    let append_time = start.elapsed();

    // -------------------------------------------------------------------------
    // 3. NEW WAY: materialize_contiguous (O(n) controlled copy)
    // -------------------------------------------------------------------------
    let manager = BlockManager::new(num_blocks, block_size, num_heads, head_dim);
    let seq_id = 0;
    for _ in 0..seq_len {
        manager.append_slot(seq_id).unwrap();
    }

    let start = Instant::now();
    let (_, _) = manager.materialize_contiguous(seq_id).unwrap();
    let materialize_time = start.elapsed();

    // -------------------------------------------------------------------------
    // Print Results
    // -------------------------------------------------------------------------
    println!("\n");
    println!("==========================================");
    println!("           BENCHMARK RESULTS");
    println!("==========================================");
    println!("| Operation                           | Time     |");
    println!("|-------------------------------------|----------|");
    println!("| Tensor::cat (O(n) reallocation)     | {:>8?} |", cat_time);
    println!("| BlockManager::append (O(1))         | {:>8?} |", append_time);
    println!("| materialize_contiguous (O(n) copy)  | {:>8?} |", materialize_time);
    println!("==========================================");

    let append_ms = append_time.as_micros() as f64 / 1000.0;
    let cat_ms = cat_time.as_micros() as f64 / 1000.0;
    let speedup = if append_ms > 0.0 { cat_ms / append_ms } else { 0.0 };

    println!("\n🚀 BlockManager::append is ~{:.1}x faster than Tensor::cat for a single append operation!", speedup);
    println!("\n💡 materialize_contiguous copies the entire cache in ~{:.2}ms (no reallocation spikes).", materialize_time.as_micros() as f64 / 1000.0);
}