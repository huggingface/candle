#![cfg(feature = "cuda")]

use candle_core::cuda_backend::cudarc::driver::{
    DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use candle_core::cuda_backend::{kernel_name, kernels, CudaDevice};
use candle_core::{DType, Result, WithDType};

// The production grid-stride limit would require trillions of input elements.
// Launch the actual production kernels with a small grid to exercise reuse of
// CUB shared storage, including a short tail followed by the next row's tile.
fn check_grid_stride<T, A>(dev: &CudaDevice, rows: usize) -> Result<()>
where
    T: WithDType + DeviceRepr + ValidAsZeroBits + std::fmt::Debug,
    A: WithDType + DeviceRepr + ValidAsZeroBits,
{
    let lowp = T::DTYPE != A::DTYPE;
    let grid_rows = rows as u32;
    let tile_size = 4096usize;
    let tile_threads = 256i32;
    for tail in [0, 1, 2, 31, 32, 33, 4095] {
        for cols in [tile_size + tail, 2 * tile_size + tail] {
            let tiles = cols.div_ceil(tile_size);
            let tasks = rows * tiles;
            let row_reciprocal = if rows == 1 {
                0
            } else {
                (1u64 << 32).div_ceil(rows as u64) as u32
            };
            let mut values = vec![T::from_f64(0.)];
            let mut expected = Vec::with_capacity(rows * cols);
            for row in 0..rows {
                let mut prefix = 0.;
                for col in 0..cols {
                    let value = ((row * 7 + col * 13) % 17) as f64
                        - if T::DTYPE == DType::U32 { 0. } else { 8. };
                    let value = T::from_f64(value / if T::DTYPE.is_int() { 1. } else { 8. });
                    values.push(value);
                    prefix += value.to_f64();
                    expected.push(T::from_f64(prefix));
                }
            }
            let allocation = dev.clone_htod(&values)?;
            let src = allocation.slice(1..values.len());
            let mut launches = vec![
                (1, "_grid_stride"),
                (2, "_grid_stride"),
                (4, "_grid_stride"),
                (tasks as u32, ""),
                (tasks as u32 + 1, "_grid_stride"),
            ];
            if rows == 1 {
                launches.push((tasks as u32, "_single_row"));
            }
            if rows > 1 && tiles == 2 {
                launches.push((tasks as u32, "_two_tiles"));
            }
            for (blocks, suffix) in launches {
                let mut dst = unsafe { dev.alloc::<T>(rows * cols)? };
                let mut totals = unsafe { dev.alloc::<A>(tasks)? };
                let mut scanned = unsafe { dev.alloc::<A>(tasks)? };
                let cfg = LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (tile_threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let entry = |name: &str| format!("{name}{suffix}");
                let first = dev.get_or_load_func(
                    &kernel_name::<T>(&entry(if lowp {
                        "cumsum_tile_totals"
                    } else {
                        "cumsum_tile"
                    })),
                    &kernels::CUMSUM,
                )?;
                let mut builder = first.builder();
                builder.arg(&src);
                if !lowp {
                    builder.arg(&mut dst);
                }
                builder.arg(&mut totals);
                builder.arg(&cols);
                if !lowp {
                    builder.arg(&tile_size);
                }
                builder.arg(&tiles);
                if !lowp {
                    builder.arg(&tile_threads);
                }
                builder.arg(&tasks);
                builder.arg(&grid_rows);
                builder.arg(&row_reciprocal);
                unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;

                let scan_threads = tiles.next_power_of_two() as i32;
                let scan_cfg = LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (scan_threads as u32, 1, 1),
                    shared_mem_bytes: scan_threads as u32 * std::mem::size_of::<A>() as u32,
                };
                let scan =
                    dev.get_or_load_func(&kernel_name::<A>("cumsum_last_dim"), &kernels::CUMSUM)?;
                let mut builder = scan.builder();
                builder.arg(&totals);
                builder.arg(&mut scanned);
                builder.arg(&tiles);
                builder.arg(&scan_threads);
                unsafe { builder.launch(scan_cfg) }.map_err(candle_core::Error::wrap)?;

                let last = dev.get_or_load_func(
                    &kernel_name::<T>(&entry(if lowp {
                        "cumsum_rescan"
                    } else {
                        "cumsum_add_offsets"
                    })),
                    &kernels::CUMSUM,
                )?;
                let mut builder = last.builder();
                if lowp {
                    builder.arg(&src);
                }
                builder.arg(&mut dst);
                builder.arg(&scanned);
                builder.arg(&cols);
                if !lowp {
                    builder.arg(&tile_size);
                }
                builder.arg(&tiles);
                builder.arg(&tasks);
                builder.arg(&grid_rows);
                builder.arg(&row_reciprocal);
                unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
                assert_eq!(
                    dev.clone_dtoh(&dst)?,
                    expected,
                    "dtype {:?}, rows {rows}, tail {tail}, blocks {blocks}, entry {suffix}",
                    T::DTYPE
                );
            }
        }
    }
    Ok(())
}

#[test]
fn cumsum_cuda_grid_stride() -> Result<()> {
    let dev = CudaDevice::new_with_stream(0)?;
    check_grid_stride::<f32, f32>(&dev, 3)?;
    check_grid_stride::<f64, f64>(&dev, 3)?;
    check_grid_stride::<u32, u32>(&dev, 3)?;
    check_grid_stride::<i64, i64>(&dev, 3)?;
    check_grid_stride::<half::f16, f32>(&dev, 3)?;
    check_grid_stride::<half::bf16, f32>(&dev, 3)?;
    Ok(())
}

#[test]
fn cumsum_cuda_single_row_entries() -> Result<()> {
    let dev = CudaDevice::new_with_stream(0)?;
    check_grid_stride::<f32, f32>(&dev, 1)?;
    check_grid_stride::<f64, f64>(&dev, 1)?;
    check_grid_stride::<u32, u32>(&dev, 1)?;
    check_grid_stride::<i64, i64>(&dev, 1)?;
    check_grid_stride::<half::f16, f32>(&dev, 1)?;
    check_grid_stride::<half::bf16, f32>(&dev, 1)?;
    Ok(())
}
