use anyhow::Result;
use candle_metal_kernels::GemmDType;
/// This example contains some simple benchmarks so that it's easy to run them in perf etc.
use clap::{Parser, Subcommand};
use half::f16;

fn run_gemm(f32: bool, n: usize) -> Result<()> {
    const WARMUP_ITERS: usize = 2;
    const MIN_DUR: f64 = 4.;

    let device = metal::Device::system_default().unwrap();

    let (b, m, n, k) = (1, n, n, n);
    let kernels = candle_metal_kernels::Kernels::new();
    let command_queue = device.new_command_queue();
    let options = metal::MTLResourceOptions::StorageModeManaged;

    let (lhs, rhs) = if f32 {
        let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
        let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
        let lhs = device.new_buffer_with_data(
            lhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(&lhs) as u64,
            options,
        );
        let rhs = device.new_buffer_with_data(
            rhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(&rhs) as u64,
            options,
        );
        (lhs, rhs)
    } else {
        let lhs: Vec<f16> = (0..b * m * k).map(|f| f16::from_f32(f as f32)).collect();
        let rhs: Vec<f16> = (0..b * n * k).map(|f| f16::from_f32(f as f32)).collect();
        let lhs = device.new_buffer_with_data(
            lhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(&lhs) as u64,
            options,
        );
        let rhs = device.new_buffer_with_data(
            rhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(&rhs) as u64,
            options,
        );
        (lhs, rhs)
    };
    let (dtype, sizeof) = if f32 {
        (GemmDType::F32, core::mem::size_of::<f32>())
    } else {
        (GemmDType::F16, core::mem::size_of::<f16>())
    };
    let output = device.new_buffer((b * m * n * sizeof) as u64, options);

    let mut sum_dt = 0f64;
    let mut iters = 0usize;
    for idx in 0.. {
        let command_buffer = command_queue.new_command_buffer();
        let start_time = std::time::Instant::now();
        candle_metal_kernels::call_mlx_gemm(
            &device,
            command_buffer,
            &kernels,
            dtype,
            (b, m, n, k),
            &[m * k, k, 1],
            0,
            &lhs,
            &[n * k, n, 1],
            0,
            &rhs,
            &output,
        )?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let dt = start_time.elapsed().as_secs_f64();
        if idx < WARMUP_ITERS {
            continue;
        }
        sum_dt += dt;
        iters += 1;
        if sum_dt > MIN_DUR {
            break;
        }
    }
    let gflops = (2 * n * n * n * iters) as f64 / (1e9 * sum_dt);
    println!("{dtype:?},      {n:6}      gflops {gflops:.0}");

    Ok(())
}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Gemm,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The benchmark to be run.
    #[command(subcommand)]
    task: Task,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.task {
        Task::Gemm => {
            for f32 in [false, true] {
                for n in [512, 1024, 2048, 4096] {
                    run_gemm(f32, n)?;
                }
            }
        }
    }
    Ok(())
}
