use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Module, Tensor};
use candle_nn::{linear_no_bias, LoraLinear, VarBuilder, VarMap};
use criterion::{criterion_group, Criterion};
use std::hint::black_box;
use std::time::Instant;

const HIDDEN: usize = 1024;
const RANK: usize = 16;
const N_ADAPTERS: usize = 4;
const BATCH: usize = 16;

fn make_lora(device: &Device) -> (LoraLinear, Vec<String>) {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let base = linear_no_bias(HIDDEN, HIDDEN, vb.pp("base")).unwrap();
    let mut lora = LoraLinear::new(base);
    let names: Vec<String> = (0..N_ADAPTERS).map(|i| format!("adapter_{i}")).collect();
    for name in names.iter() {
        lora.load_adapter(name, vb.pp(name), RANK, 16.0).unwrap();
    }
    (lora, names)
}

/// Round-robin over the adapters with every fourth row left on the base
/// layer, so the batch is genuinely heterogeneous.
fn make_assignments(names: &[String]) -> Vec<Option<&str>> {
    (0..BATCH)
        .map(|i| {
            if i % 4 == 3 {
                None
            } else {
                Some(names[i % names.len()].as_str())
            }
        })
        .collect()
}

/// The host-side loop the batched path replaces: group rows by adapter, run
/// one forward per group, then scatter the group outputs back to their rows.
fn sequential_grouped_forward(
    lora: &LoraLinear,
    x: &Tensor,
    assignments: &[Option<&str>],
) -> Tensor {
    let device = x.device();
    let mut lora = lora.clone();
    let mut groups: Vec<(Option<&str>, Vec<u32>)> = Vec::new();
    for (i, assignment) in assignments.iter().enumerate() {
        match groups.iter_mut().find(|(name, _)| name == assignment) {
            Some((_, rows)) => rows.push(i as u32),
            None => groups.push((*assignment, vec![i as u32])),
        }
    }
    let mut outputs = Vec::with_capacity(groups.len());
    let mut order = Vec::with_capacity(assignments.len());
    for (name, rows) in groups {
        lora.set_active_adapter(name).unwrap();
        let idx = Tensor::from_vec(rows.clone(), rows.len(), device).unwrap();
        outputs.push(lora.forward(&x.index_select(&idx, 0).unwrap()).unwrap());
        order.extend(rows);
    }
    let mut inverse = vec![0u32; order.len()];
    for (pos, row) in order.iter().enumerate() {
        inverse[*row as usize] = pos as u32;
    }
    let inverse = Tensor::from_vec(inverse, order.len(), device).unwrap();
    Tensor::cat(&outputs, 0)
        .unwrap()
        .index_select(&inverse, 0)
        .unwrap()
}

fn run_lora_benchmark(c: &mut Criterion, device: &Device, seq_len: usize) {
    let (lora, names) = make_lora(device);
    let assignments = make_assignments(&names);
    let x = Tensor::randn(0f32, 1., (BATCH, seq_len, HIDDEN), device).unwrap();

    let mut group = c.benchmark_group(device.bench_name(format!("lora_multi_adapter_s{seq_len}")));
    {
        let (lora, x, assignments) = (lora.clone(), x.clone(), assignments.clone());
        group.bench_function("batched", move |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(
                        lora.forward_with_adapters(black_box(&x), black_box(&assignments))
                            .unwrap(),
                    );
                }
                x.device().sync().unwrap();
                start.elapsed()
            })
        });
    }
    group.bench_function("sequential", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(sequential_grouped_forward(
                    black_box(&lora),
                    black_box(&x),
                    black_box(&assignments),
                ));
            }
            x.device().sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = BenchDeviceHandler::new().unwrap();
    for d in device.devices {
        // Decode-style (one token per sequence) and prefill-style batches.
        run_lora_benchmark(c, &d, 1);
        run_lora_benchmark(c, &d, 128);
    }
}

criterion_group!(benches, criterion_benchmark);
