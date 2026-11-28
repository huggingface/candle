use candle::{Device, DType, Tensor};
use candle_transformers::kv_cache::{PagedCacheConfig, PagedKvCache};
use std::time::Instant;

/// Single-layer concat cache for fair baseline comparison.
struct ConcatKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    dim: usize,
}

impl ConcatKvCache {
    fn new(dim: usize) -> Self {
        Self { k: None, v: None, dim }
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> candle::Result<(Tensor, Tensor)> {
        let k = k.contiguous()?.detach();
        let v = v.contiguous()?.detach();
        self.k = Some(match &self.k {
            None => k,
            Some(cache) => Tensor::cat(&[cache, &k], self.dim)?.detach(),
        });
        self.v = Some(match &self.v {
            None => v,
            Some(cache) => Tensor::cat(&[cache, &v], self.dim)?.detach(),
        });
        Ok((
            self.k.as_ref().unwrap().clone(),
            self.v.as_ref().unwrap().clone(),
        ))
    }
}

#[test]
fn test_paged_cache_roundtrip() -> candle::Result<()> {
    let device = Device::Cpu;
    let config = PagedCacheConfig {
        num_blocks: 16,
        block_size: 64,
        num_layers: 2,
        num_kv_heads: 4,
        head_dim: 32,
        dtype: DType::F32,
        device: device.clone(),
    };

    let mut cache = PagedKvCache::new(config)?;
    let seq_id = cache.add_sequence();

    let k1 = Tensor::randn(0f32, 1f32, (1, 4, 1, 32), &device)?;
    let v1 = Tensor::randn(0f32, 1f32, (1, 4, 1, 32), &device)?;

    for _ in 0..3 {
        cache.append_kv(seq_id, 0, &k1, &v1)?;
    }

    let (k_out, v_out) = cache.gather_kv(seq_id, 0)?;
    assert_eq!(k_out.dims(), &[3, 4, 32]);
    assert_eq!(v_out.dims(), &[3, 4, 32]);

    Ok(())
}

#[test]
fn test_paged_cache_cow_fork() -> candle::Result<()> {
    let device = Device::Cpu;
    let config = PagedCacheConfig {
        num_blocks: 16,
        block_size: 64,
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 16,
        dtype: DType::F32,
        device: device.clone(),
    };

    let mut cache = PagedKvCache::new(config)?;
    let seq_a = cache.add_sequence();

    let k = Tensor::ones((1, 2, 1, 16), DType::F32, &device)?;
    let v = Tensor::ones((1, 2, 1, 16), DType::F32, &device)?;

    cache.append_kv(seq_a, 0, &k, &v)?;
    cache.append_kv(seq_a, 0, &k, &v)?;

    let seq_b = cache.fork_sequence(seq_a)?;

    assert_eq!(cache.num_free_blocks(0), 16 - 1);

    cache.append_kv(seq_b, 0, &k, &v)?;

    let (k_a, _) = cache.gather_kv(seq_a, 0)?;
    let (k_b, _) = cache.gather_kv(seq_b, 0)?;
    assert_eq!(k_a.dims()[0], 2);
    assert_eq!(k_b.dims()[0], 3);

    Ok(())
}

#[test]
fn test_paged_cache_index_caching() -> candle::Result<()> {
    let device = Device::Cpu;
    let config = PagedCacheConfig {
        num_blocks: 16,
        block_size: 4,
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 8,
        dtype: DType::F32,
        device: device.clone(),
    };

    let mut cache = PagedKvCache::new(config)?;
    let seq_id = cache.add_sequence();
    let k = Tensor::zeros((1, 2, 1, 8), DType::F32, &device)?;
    let v = Tensor::zeros((1, 2, 1, 8), DType::F32, &device)?;

    for _ in 0..5 {
        cache.append_kv(seq_id, 0, &k, &v)?;
        let _ = cache.gather_kv(seq_id, 0)?;
    }

    assert_eq!(cache.num_blocks(seq_id, 0), 2);

    Ok(())
}

/// FAIR benchmark: single layer, same work on both sides.
#[test]
fn bench_paged_vs_concat_single_layer() -> candle::Result<()> {
    let device = Device::Cpu;
    let k_shape = (1, 32, 1, 128);
    let v_shape = (1, 32, 1, 128);
    let num_layers = 1;
    let prefill_len = 512;
    let decode_steps = 100;

    let k = Tensor::zeros(k_shape, DType::F32, &device)?;
    let v = Tensor::zeros(v_shape, DType::F32, &device)?;

    // --- ConcatKvCache baseline (1 layer) ---
    let mut concat = ConcatKvCache::new(2);
    for _ in 0..prefill_len {
        concat.append(&k, &v)?;
    }

    let t0 = Instant::now();
    for _ in 0..decode_steps {
        concat.append(&k, &v)?;
    }
    let concat_time = t0.elapsed();
    println!("ConcatKvCache (1 layer, {} decodes): {:?}", decode_steps, concat_time);

    // --- PagedKvCache (1 layer) ---
    let config = PagedCacheConfig {
        num_blocks: 64,
        block_size: 64,
        num_layers,
        num_kv_heads: 32,
        head_dim: 128,
        dtype: DType::F32,
        device: device.clone(),
    };
    let mut paged = PagedKvCache::new(config)?;
    let seq_id = paged.add_sequence();

    for _ in 0..prefill_len {
        paged.append_kv(seq_id, 0, &k, &v)?;
    }

    let t0 = Instant::now();
    for _ in 0..decode_steps {
        paged.append_kv(seq_id, 0, &k, &v)?;
    }
    let paged_time = t0.elapsed();
    println!("PagedKvCache   (1 layer, {} decodes): {:?}", decode_steps, paged_time);

    let overhead = (paged_time.as_secs_f64() / concat_time.as_secs_f64() - 1.0) * 100.0;
    println!("Paged overhead (single layer): {:.1}%", overhead);

    assert!(overhead < 80.0, "Paged cache too slow: {:.1}% overhead", overhead);

    Ok(())
}

/// FAIR benchmark: 32 layers, both sides do 32 layers.
#[test]
fn bench_paged_vs_concat_full_model() -> candle::Result<()> {
    let device = Device::Cpu;
    let k_shape = (1, 32, 1, 128);
    let v_shape = (1, 32, 1, 128);
    let num_layers = 32;
    let prefill_len = 512;
    let decode_steps = 10;

    let k = Tensor::zeros(k_shape, DType::F32, &device)?;
    let v = Tensor::zeros(v_shape, DType::F32, &device)?;

    // --- Baseline: 32 independent ConcatKvCaches ---
    let mut concat_caches: Vec<ConcatKvCache> =
        (0..num_layers).map(|_| ConcatKvCache::new(2)).collect();

    for layer in 0..num_layers {
        for _ in 0..prefill_len {
            concat_caches[layer].append(&k, &v)?;
        }
    }

    let t0 = Instant::now();
    for _ in 0..decode_steps {
        for layer in 0..num_layers {
            concat_caches[layer].append(&k, &v)?;
        }
    }
    let concat_time = t0.elapsed();
    println!("ConcatKvCache (32 layers, {} decodes): {:?}", decode_steps, concat_time);

    // --- Paged: 32 layers ---
    let config = PagedCacheConfig {
        num_blocks: 128,
        block_size: 64,
        num_layers,
        num_kv_heads: 32,
        head_dim: 128,
        dtype: DType::F32,
        device: device.clone(),
    };
    let mut paged = PagedKvCache::new(config)?;
    let seq_id = paged.add_sequence();

    for layer in 0..num_layers {
        for _ in 0..prefill_len {
            paged.append_kv(seq_id, layer, &k, &v)?;
        }
    }

    let t0 = Instant::now();
    for _ in 0..decode_steps {
        for layer in 0..num_layers {
            paged.append_kv(seq_id, layer, &k, &v)?;
        }
    }
    let paged_time = t0.elapsed();
    println!("PagedKvCache   (32 layers, {} decodes): {:?}", decode_steps, paged_time);

    let overhead = (paged_time.as_secs_f64() / concat_time.as_secs_f64() - 1.0) * 100.0;
    println!("Paged overhead (32 layers): {:.1}%", overhead);

    Ok(())
}

/// COW fork benchmark.
#[test]
fn bench_cow_fork() -> candle::Result<()> {
    let device = Device::Cpu;
    let k = Tensor::zeros((1, 32, 1, 128), DType::F32, &device)?;
    let v = Tensor::zeros((1, 32, 1, 128), DType::F32, &device)?;
    let num_layers = 32;
    let seq_len = 4096;
    let block_size = 64;

    // Per layer: 4096 / 64 = 64 blocks needed. Add slack.
    let config = PagedCacheConfig {
        num_blocks: 128,
        block_size,
        num_layers,
        num_kv_heads: 32,
        head_dim: 128,
        dtype: DType::F32,
        device: device.clone(),
    };
    let mut cache = PagedKvCache::new(config)?;
    let seq_id = cache.add_sequence();

    // Pre-fill all layers
    for layer in 0..num_layers {
        for _ in 0..seq_len {
            cache.append_kv(seq_id, layer, &k, &v)?;
        }
    }

    let t0 = Instant::now();
    for _ in 0..100 {
        cache.fork_sequence(seq_id)?;
    }
    let elapsed = t0.elapsed();
    println!("100 COW forks at {} ctx ({} layers): {:?}", seq_len, num_layers, elapsed);
    println!("Per fork: {:?}", elapsed / 100);

    assert!(elapsed.as_millis() < 50, "COW fork too slow: {:?}", elapsed);

    Ok(())
}
/// Honest benchmark: write + gather at SHORT sequence lengths where fast-path applies.
#[test]
fn bench_paged_write_and_gather_short() -> candle::Result<()> {
    let device = Device::Cpu;
    let k_shape = (1, 32, 1, 128);
    let v_shape = (1, 32, 1, 128);
    let num_layers = 32;
    let seq_len = 512; // Short sequence: fits in 8 blocks of 64

    let k = Tensor::zeros(k_shape, DType::F32, &device)?;
    let v = Tensor::zeros(v_shape, DType::F32, &device)?;

    // --- Paged ---
    let config = PagedCacheConfig {
        num_blocks: 128,
        block_size: 64,
        num_layers,
        num_kv_heads: 32,
        head_dim: 128,
        dtype: DType::F32,
        device: device.clone(),
    };
    let mut paged = PagedKvCache::new(config)?;
    let seq_id = paged.add_sequence();

    for layer in 0..num_layers {
        for _ in 0..seq_len {
            paged.append_kv(seq_id, layer, &k, &v)?;
        }
    }

    let t0 = Instant::now();
    for layer in 0..num_layers {
        paged.append_kv(seq_id, layer, &k, &v)?;
        let _ = paged.gather_kv(seq_id, layer)?;
    }
    let paged_time = t0.elapsed();
    println!(
        "PagedKvCache write+gather (32 layers, seq={}): {:?}",
        seq_len + 1,
        paged_time
    );

    // --- Concat baseline ---
    let mut concat_caches: Vec<ConcatKvCache> =
        (0..num_layers).map(|_| ConcatKvCache::new(2)).collect();

    for layer in 0..num_layers {
        for _ in 0..seq_len {
            concat_caches[layer].append(&k, &v)?;
        }
    }

    let t0 = Instant::now();
    for layer in 0..num_layers {
        concat_caches[layer].append(&k, &v)?;
    }
    let concat_time = t0.elapsed();
    println!(
        "ConcatKvCache write+access (32 layers, seq={}): {:?}",
        seq_len + 1,
        concat_time
    );

    let overhead = (paged_time.as_secs_f64() / concat_time.as_secs_f64() - 1.0) * 100.0;
    println!("Paged overhead at seq=512: {:.1}%", overhead);

    // At 512 tokens, fast-path may trigger (single block = 64 tokens... wait, 512/64 = 8 blocks)
    // So fast-path does NOT trigger. This is the honest software gather cost at medium length.
    // Expect 200-500% overhead. That is acceptable information.
    Ok(())
}

/// Honest benchmark: write + gather at VERY short lengths where fast-path (narrow) applies.
#[test]
fn bench_paged_write_and_gather_fast_path() -> candle::Result<()> {
    let device = Device::Cpu;
    let k_shape = (1, 32, 1, 128);
    let v_shape = (1, 32, 1, 128);
    let num_layers = 32;
    let seq_len = 32; // Fits in ONE block of 64 -> fast path triggers

    let k = Tensor::zeros(k_shape, DType::F32, &device)?;
    let v = Tensor::zeros(v_shape, DType::F32, &device)?;

    let config = PagedCacheConfig {
        num_blocks: 128,
        block_size: 64,
        num_layers,
        num_kv_heads: 32,
        head_dim: 128,
        dtype: DType::F32,
        device: device.clone(),
    };
    let mut paged = PagedKvCache::new(config)?;
    let seq_id = paged.add_sequence();

    for layer in 0..num_layers {
        for _ in 0..seq_len {
            paged.append_kv(seq_id, layer, &k, &v)?;
        }
    }

    let t0 = Instant::now();
    for layer in 0..num_layers {
        paged.append_kv(seq_id, layer, &k, &v)?;
        let _ = paged.gather_kv(seq_id, layer)?;
    }
    let paged_time = t0.elapsed();

    let mut concat = ConcatKvCache::new(2);
    for _ in 0..seq_len {
        concat.append(&k, &v)?;
    }

    let t0 = Instant::now();
    concat.append(&k, &v)?;
    let concat_time = t0.elapsed();

    let overhead = (paged_time.as_secs_f64() / concat_time.as_secs_f64() - 1.0) * 100.0;
    println!("Paged overhead at seq=32 (fast-path): {:.1}%", overhead);

    // Fast-path should be near zero overhead or even faster because narrow is free
    Ok(())
}