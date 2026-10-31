use super::*;
use crate::metal::{Commands, ResidencySet};
use core::ffi::c_void;
use half::{bf16, f16};
use rand::prelude::SliceRandom;
use rand::{rng, Rng};
use std::sync::Arc;
use std::thread;

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    Commands::new(queue, &residency_set).unwrap()
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

fn new_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let options = RESOURCE_OPTIONS;
    let ptr = data.as_ptr() as *const c_void;
    let size = std::mem::size_of_val(data);
    device.new_buffer_with_data(ptr, size, options).unwrap()
}

fn device() -> Device {
    Device::system_default().unwrap()
}

#[test]
fn pipeline_cache_distinguishes_sources() {
    let device = device();
    let kernels = Kernels::new();

    // Prime the cache with a name that is not present in the binary library.
    kernels
        .load_pipeline(&device, Source::Unary, "cos_f32")
        .unwrap();
    assert!(matches!(
        kernels.load_pipeline(&device, Source::Binary, "cos_f32"),
        Err(MetalKernelError::LoadFunctionError(_))
    ));
}

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

fn approx_f16(v: Vec<f16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn approx_bf16(v: Vec<bf16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn run<T: Clone>(v: &[T], name: unary::contiguous::Kernel) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let input = BufferOffset {
        buffer: &input,
        offset_in_bytes: 0,
    };
    let output = new_buffer(&device, v);
    call_unary_contiguous(
        &device,
        &encoder,
        &kernels,
        name,
        size_of::<T>(),
        v.len(),
        input,
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, v.len())
}

fn run_binary<T: Clone, S: ToString>(x: &[T], y: &[T], name: S) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;
    let left = new_buffer(&device, x);
    let right = new_buffer(&device, y);
    let output = device
        .new_buffer(std::mem::size_of_val(x), options)
        .unwrap();
    call_binary_contiguous(
        &device,
        &encoder,
        &kernels,
        name,
        size_of::<T>(),
        x.len(),
        BufferOffset::zero_offset(&left),
        BufferOffset::zero_offset(&right),
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, x.len())
}

fn run_strided<T: Clone>(
    v: &[T],
    kernel: unary::strided::Kernel,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Vec<T> {
    let device = device();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let input = BufferOffset {
        buffer: &input,
        offset_in_bytes: offset,
    };
    let output_b = new_buffer(&device, v);
    let output = BufferOffset {
        buffer: &output_b,
        offset_in_bytes: 0,
    };
    let kernels = Kernels::new();
    call_unary_strided(
        &device, &encoder, &kernels, kernel, shape, input, strides, output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output_b, v.len())
}

#[test]
fn cos_f32() {
    let v = vec![1.0f32, 2.0, 3.0];
    let results = run(&v, unary::contiguous::cos::FLOAT);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403, -0.4161, -0.99]);
    assert_eq!(approx(expected, 4), vec![0.5403, -0.4161, -0.99]);

    let v = vec![1.0f32; 10_000];
    let results = run(&v, unary::contiguous::cos::FLOAT);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
    assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
}

#[test]
fn cos_f32_strided() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![6];
    let strides = vec![1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Contiguous
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Transposed
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![1, 3];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.6536, -0.4161, 0.2837, -0.99, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Very large
    let v = vec![1.0f32; 10_000];
    let shape = vec![2, 5_000];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
    assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
}

#[test]
fn cos_strided_random() {
    let v: Vec<_> = (0..10_000).map(|_| rand::random::<f32>()).collect();
    let shape = vec![5_000, 2];
    let strides = vec![1, 5_000];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(vec![results[0]], 4), approx(vec![expected[0]], 4));
    assert_eq!(
        approx(vec![results[1]], 4),
        approx(vec![expected[5_000]], 4)
    );
    assert_eq!(approx(vec![results[2]], 4), approx(vec![expected[1]], 4));
    assert_eq!(
        approx(vec![results[3]], 4),
        approx(vec![expected[5_001]], 4)
    );
    assert_eq!(
        approx(vec![results[5_000]], 4),
        approx(vec![expected[2_500]], 4)
    );
}

#[test]
fn gelu_f16() {
    let v: Vec<f16> = [-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let expected: Vec<f32> = vec![-0.0, -0.159, 0.0, 0.841, 1.954, 2.996, 10.0, 20.0];
    let results = run(&v, unary::contiguous::gelu::HALF);
    assert_eq!(approx_f16(results, 3), expected);
}

#[test]
fn gelu_f32() {
    let v: Vec<f32> = vec![-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0];
    let expected: Vec<f32> = vec![-0.0, -0.159, 0.0, 0.841, 1.955, 2.996, 10.0, 20.0];
    let results = run(&v, unary::contiguous::gelu::FLOAT);
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn silu_f16() {
    let v: Vec<f16> = [-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let expected: Vec<f32> = vec![-0.0, -0.27, 0.0, 0.73, 1.76, 2.86, 10.0, 20.0];
    let results = run(&v, unary::contiguous::silu::HALF);
    assert_eq!(approx_f16(results, 2), expected);
}

#[test]
fn silu_f32() {
    let v: Vec<f32> = vec![-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0];
    let expected: Vec<f32> = vec![-0.0, -0.269, 0.0, 0.731, 1.762, 2.858, 10.0, 20.0];
    let results = run(&v, unary::contiguous::silu::FLOAT);
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn binary_add_f32() {
    let left = vec![1.0f32, 2.0, 3.0];
    let right = vec![2.0f32, 3.1, 4.2];
    let results = run_binary(&left, &right, "badd_f32");
    let expected: Vec<_> = left
        .iter()
        .zip(right.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    assert_eq!(approx(results, 4), vec![3.0f32, 5.1, 7.2]);
    assert_eq!(approx(expected, 4), vec![3.0f32, 5.1, 7.2]);
}

#[test]
fn binary_ops_bf16() {
    let lhs: Vec<bf16> = [1.1f32, 2.2, 3.3].into_iter().map(bf16::from_f32).collect();
    let rhs: Vec<bf16> = [4.2f32, 5.5f32, 6.91f32]
        .into_iter()
        .map(bf16::from_f32)
        .collect();

    macro_rules! binary_op {
        ($opname:ident, $dtype:ident, $opexpr:expr) => {{
            let results = run_binary(
                &lhs,
                &rhs,
                concat!(stringify!($opname), "_", stringify!($dtype)),
            );
            let expected: Vec<bf16> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&$dtype, &$dtype)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }
    binary_op!(badd, bf16, |x, y| x + y);
    binary_op!(bsub, bf16, |x, y| x - y);
    binary_op!(bmul, bf16, |x, y| x * y);
    binary_op!(bdiv, bf16, |x, y| x / y);
    binary_op!(bminimum, bf16, |x: bf16, y| x.min(y));
    binary_op!(bmaximum, bf16, |x: bf16, y| x.max(y));
}

fn run_cast<T: Clone, U: Clone>(v: &[T], name: &'static str) -> Vec<U> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let options = RESOURCE_OPTIONS;
    let size = v.len() * std::mem::size_of::<U>();
    let output = device.new_buffer(size, options).unwrap();

    call_cast_contiguous(
        &device,
        &encoder,
        &kernels,
        name,
        size_of::<T>(),
        v.len(),
        BufferOffset::zero_offset(&input),
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, v.len())
}

#[test]
fn cast_f32() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // f32 -> f16
    let results: Vec<half::f16> = run_cast(&v_f32, "cast_f32_f16");
    assert_eq!(results, v_f16);

    // f32 -> bf16
    let results: Vec<bf16> = run_cast(&v_f32, "cast_f32_bf16");
    assert_eq!(results, v_bf16);

    // f32 -> u32
    let results: Vec<u32> = run_cast(&v_f32, "cast_f32_u32");
    assert_eq!(results, v_u32);

    // f32 -> u8
    let results: Vec<u8> = run_cast(&v_f32, "cast_f32_u8");
    assert_eq!(results, v_u8);

    // f32 -> i64
    let results: Vec<i64> = run_cast(&v_f32, "cast_f32_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_f16() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // f16 -> f32
    let results: Vec<f32> = run_cast(&v_f16, "cast_f16_f32");
    assert_eq!(results, v_f32);

    // f16 -> bf16
    let results: Vec<bf16> = run_cast(&v_f16, "cast_f16_bf16");
    assert_eq!(results, v_bf16);

    // f16 -> u32
    let results: Vec<u32> = run_cast(&v_f16, "cast_f16_u32");
    assert_eq!(results, v_u32);

    // f16 -> u8
    let results: Vec<u8> = run_cast(&v_f16, "cast_f16_u8");
    assert_eq!(results, v_u8);

    // f16 -> i64
    let results: Vec<i64> = run_cast(&v_f16, "cast_f16_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_bf16() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // bf16 -> f32
    let results: Vec<f32> = run_cast(&v_bf16, "cast_bf16_f32");
    assert_eq!(results, v_f32);

    // bf16 -> f16
    let results: Vec<f16> = run_cast(&v_bf16, "cast_bf16_f16");
    assert_eq!(results, v_f16);

    // bf16 -> u32
    let results: Vec<u32> = run_cast(&v_bf16, "cast_bf16_u32");
    assert_eq!(results, v_u32);

    // bf16 -> u8
    let results: Vec<u8> = run_cast(&v_bf16, "cast_bf16_u8");
    assert_eq!(results, v_u8);

    // bf16 -> i64
    let results: Vec<i64> = run_cast(&v_bf16, "cast_bf16_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_u32() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // u32 -> f32
    let results: Vec<f32> = run_cast(&v_u32, "cast_u32_f32");
    assert_eq!(results, v_f32);

    // u32 -> f16
    let results: Vec<f16> = run_cast(&v_u32, "cast_u32_f16");
    assert_eq!(results, v_f16);

    // u32 -> bf16
    let results: Vec<bf16> = run_cast(&v_u32, "cast_u32_bf16");
    assert_eq!(results, v_bf16);

    // u32 -> u8
    let results: Vec<u8> = run_cast(&v_u32, "cast_u32_u8");
    assert_eq!(results, v_u8);

    // u32 -> i64
    let results: Vec<i64> = run_cast(&v_u32, "cast_u32_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_u8() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // u8 -> f32
    let results: Vec<f32> = run_cast(&v_u8, "cast_u8_f32");
    assert_eq!(results, v_f32);

    // u8 -> f16
    let results: Vec<f16> = run_cast(&v_u8, "cast_u8_f16");
    assert_eq!(results, v_f16);

    // u8 -> bf16
    let results: Vec<bf16> = run_cast(&v_u8, "cast_u8_bf16");
    assert_eq!(results, v_bf16);

    // u8 -> u32
    let results: Vec<u32> = run_cast(&v_u8, "cast_u8_u32");
    assert_eq!(results, v_u32);

    // u8 -> i64
    let results: Vec<i64> = run_cast(&v_u8, "cast_u8_i64");
    assert_eq!(results, v_i64);
}

#[test]
fn cast_i64() {
    let v_f64 = [1.0f64, 2.0, 3.0];
    let v_f32: Vec<f32> = v_f64.iter().map(|&v| v as f32).collect();
    let v_f16: Vec<f16> = v_f64.iter().map(|&v| f16::from_f32(v as f32)).collect();
    let v_bf16: Vec<bf16> = v_f64.iter().map(|&v| bf16::from_f32(v as f32)).collect();
    let v_u32: Vec<u32> = v_f64.iter().map(|&v| v as u32).collect();
    let v_u8: Vec<u8> = v_f64.iter().map(|&v| v as u8).collect();
    let v_i64: Vec<i64> = v_f64.iter().map(|&v| v as i64).collect();

    // i64 -> f32
    let results: Vec<f32> = run_cast(&v_i64, "cast_i64_f32");
    assert_eq!(results, v_f32);

    // i64 -> f16
    let results: Vec<f16> = run_cast(&v_i64, "cast_i64_f16");
    assert_eq!(results, v_f16);

    // i64 -> bf16
    let results: Vec<bf16> = run_cast(&v_i64, "cast_i64_bf16");
    assert_eq!(results, v_bf16);

    // i64 -> u32
    let results: Vec<u32> = run_cast(&v_i64, "cast_i64_u32");
    assert_eq!(results, v_u32);

    // i64 -> u8
    let results: Vec<u8> = run_cast(&v_i64, "cast_i64_u8");
    assert_eq!(results, v_u8);
}

fn run_affine<T: Clone>(v: &[T], mul: f64, add: f64) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    let size = v.len();

    call_affine(
        &device,
        &encoder,
        &kernels,
        "affine_f32",
        size_of::<T>(),
        size,
        BufferOffset::zero_offset(&input),
        &output,
        mul as f32,
        add as f32,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, v.len())
}

fn run_affine_strided<T: Clone>(
    v: &[T],
    shape: &[usize],
    strides: &[usize],
    mul: f64,
    add: f64,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    call_affine_strided(
        &device,
        &encoder,
        &kernels,
        "affine_f32_strided",
        shape,
        BufferOffset::zero_offset(&input),
        strides,
        &output,
        mul as f32,
        add as f32,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let len: usize = shape.iter().product();
    read_to_vec(&output, len)
}

#[test]
fn affine() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mul = 1.5;
    let add = 1.1;
    let result = run_affine(&input, mul, add);
    assert_eq!(result, vec![2.6, 4.1, 5.6, 7.1, 8.6, 10.1, 11.6, 13.1]);

    let input = [1.0f32; 40_000];
    let mul = 1.5;
    let add = 1.1;
    let result = run_affine(&input, mul, add);
    assert_eq!(result, vec![2.6; 40_000]);
}

#[test]
fn affine_strided() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mul = 1.5;
    let add = 1.1;
    let shape = [4];
    let strides = [2];
    let result = run_affine_strided(&input, &shape, &strides, mul, add);
    // 1 on 2
    assert_eq!(result, vec![2.6, 5.6, 8.6, 11.6]);
}

fn run_mlx_sort<T: Clone>(v: &[T], ncols: usize) -> Vec<u32> {
    let nrows = v.len() / ncols;
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let input = new_buffer(&device, v);
    let indexes = vec![0u32; v.len()];
    let output = new_buffer(&device, &indexes);

    call_mlx_arg_sort(
        &device,
        &encoder,
        &kernels,
        DType::F32,
        nrows,
        ncols,
        BufferOffset::zero_offset(&input),
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, v.len())
}

#[test]
fn mlx_sort() {
    use rand::SeedableRng;
    use rand_distr::Distribution;

    let input: Vec<_> = (0..8).map(|v| v as f32).collect();
    let result = run_mlx_sort(&input, 4);
    assert_eq!(result, [0, 1, 2, 3, 0, 1, 2, 3]);
    let input: Vec<_> = (0..8).rev().map(|v| v as f32).collect();
    let result = run_mlx_sort(&input, 4);
    assert_eq!(result, [3, 2, 1, 0, 3, 2, 1, 0]);
    let input: Vec<_> = (0..1000).rev().map(|v| v as f32).collect();
    let result = run_mlx_sort(&input, 200);
    let out: Vec<_> = (0..200).rev().collect();
    assert_eq!(&result[..200], out);
    assert_eq!(&result[200..400], out);
    assert_eq!(&result[400..600], out);
    assert_eq!(&result[600..800], out);
    assert_eq!(&result[800..], out);

    // Multi-block test
    let ncols = 16000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(299792458);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let input: Vec<f32> = (0..ncols * 16).map(|_| normal.sample(&mut rng)).collect();
    let result = run_mlx_sort(&input, ncols);
    for start in 0..16 {
        let slice = &input[start * ncols..(start + 1) * ncols];
        let result = &result[start * ncols..(start + 1) * ncols];
        let mut perm: Vec<usize> = (0..ncols).collect();
        perm.sort_by(|i1, i2| slice[*i1].total_cmp(&slice[*i2]));
        let perm: Vec<_> = perm.into_iter().map(|v| v as u32).collect();
        assert_eq!(perm, result);
    }
}

#[test]
fn index_select() {
    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(result, vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]);

    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [2, 5];
    let stride = [1, 2];
    let ids = [0u32, 1, 0];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0f32, 2.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn index_select_strided() {
    let embedding = (0..16).map(|x| x as f32).collect::<Vec<_>>();
    let shape = [2, 2];
    let stride = [2, 4];
    let ids = [0u32];
    let dim = 0;
    let result = run_index_select_strided(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(result, vec![0.0, 4.0]);
}

#[test]
fn index_select_f16() {
    let embedding: Vec<_> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f16");
    assert_eq!(
        approx_f16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u32_bf16() {
    let embedding: Vec<bf16> = (1..=10).map(|x| bf16::from_f32(x as f32)).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_bf16");
    assert_eq!(
        approx_bf16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u8_bf16() {
    let embedding: Vec<bf16> = (1..=10).map(|x| bf16::from_f32(x as f32)).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u8, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u8_bf16");
    assert_eq!(
        approx_bf16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u32_i64() {
    let embedding: Vec<i64> = (1..=10).map(|x| x as i64).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_i64");
    assert_eq!(result, vec![1i64, 2, 9, 10, 5, 6]);
}

#[test]
fn index_select_is_u8_i64() {
    let embedding: Vec<i64> = (1..=10).map(|x| x as i64).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u8, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u8_i64");
    assert_eq!(result, vec![1i64, 2, 9, 10, 5, 6]);
}

#[test]
fn index_select_is_i64_i64() {
    let embedding: Vec<i64> = (1..=10).map(|x| x as i64).collect();
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0i64, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_i64_i64");
    assert_eq!(result, vec![1i64, 2, 9, 10, 5, 6]);
}

#[test]
fn index_select_dim1() {
    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let stride = [2, 1];
    let ids = [0u32, 1, 0];
    let dim = 1;
    let result = run_index_select(&embedding, &shape, &stride, &ids, dim, "is_u32_f32");
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0, 6.0, 5.0, 7.0, 8.0f32, 7.0, 9.0, 10.0, 9.0]
    );
}

fn run_index_select<T: Clone, I: Clone + std::fmt::Debug>(
    embeddings: &[T],
    shape: &[usize],
    stride: &[usize],
    ids: &[I],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = Device::system_default().expect("no device found");

    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let embeddings_buffer = new_buffer(&device, embeddings);
    let ids_buffer = new_buffer(&device, ids);

    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let dst_el = ids.len() * left_size * right_size;
    let dst_buffer = new_buffer(&device, &vec![0.0f32; dst_el]);

    let kernels = Kernels::new();
    call_index_select(
        &device,
        &encoder,
        &kernels,
        name,
        shape,
        ids.len(),
        dim,
        true,
        shape,
        stride,
        BufferOffset::zero_offset(&embeddings_buffer),
        BufferOffset::zero_offset(&ids_buffer),
        &dst_buffer,
    )
    .unwrap();

    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&dst_buffer, dst_el)
}

fn run_index_select_strided<T: Clone, I: Clone + std::fmt::Debug>(
    embeddings: &[T],
    shape: &[usize],
    stride: &[usize],
    ids: &[I],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = Device::system_default().expect("no device found");

    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let embeddings_buffer = new_buffer(&device, embeddings);
    let ids_buffer = new_buffer(&device, ids);

    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let dst_el = ids.len() * left_size * right_size;
    let dst_buffer = new_buffer(&device, &vec![0.0f32; dst_el]);

    let kernels = Kernels::new();
    call_index_select(
        &device,
        &encoder,
        &kernels,
        name,
        shape,
        ids.len(),
        dim,
        false,
        shape,
        stride,
        BufferOffset::zero_offset(&embeddings_buffer),
        BufferOffset::zero_offset(&ids_buffer),
        &dst_buffer,
    )
    .unwrap();

    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&dst_buffer, dst_el)
}

#[test]
fn cos_f16() {
    let v: Vec<f16> = [1.0f32, 2.0, 3.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let results = run(&v, unary::contiguous::cos::HALF);
    let expected: Vec<f16> = v.iter().map(|v| f16::from_f32(v.to_f32().cos())).collect();
    assert_eq!(approx_f16(results, 2), vec![0.54, -0.42, -0.99]);
    assert_eq!(approx_f16(expected, 2), vec![0.54, -0.42, -0.99]);
}

fn run_reduce<T, U: Clone>(
    v: &[T],
    in_length: usize,
    out_length: usize,
    name: &'static str,
) -> Vec<U> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);

    let options = RESOURCE_OPTIONS;
    let output = device
        .new_buffer(out_length * core::mem::size_of::<U>(), options)
        .unwrap();
    let shape = vec![in_length];
    match call_reduce_contiguous(
        &device,
        &encoder,
        &kernels,
        name,
        &shape,
        out_length,
        BufferOffset::zero_offset(&input),
        &output,
    ) {
        Ok(_) => {}
        Err(e) => {
            println!("{e}");
            panic!();
        }
    }
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, out_length)
}

fn run_softmax<T: Clone + std::fmt::Debug>(v: &[T], last_dim: usize, name: &'static str) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);
    call_last_softmax(
        &device,
        &encoder,
        &kernels,
        name,
        v.len(),
        last_dim,
        &input,
        0,
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, v.len())
}

const fn create_array<const N: usize>() -> [f32; N] {
    let mut array: [f32; N] = [0.0; N];
    let mut i = 1;
    while i <= N {
        array[i - 1] = i as f32;
        i += 1;
    }
    array
}

const fn correct_sum<const N: usize, const D: usize>() -> [f32; D] {
    let mut sum = 0;
    let mut results: [f32; D] = [0.0; D];
    let mut i = 1;
    let mut j = 1;
    while i <= N {
        sum += i;
        i += 1;
        if i > j * N / D {
            results[j - 1] = sum as f32;
            j += 1;
            sum = 0;
        }
    }
    results
}

const fn correct_max<const N: usize, const D: usize>() -> [f32; D] {
    let mut results: [f32; D] = [0.0; D];
    let mut i = 1;
    let mut j = 1;
    while i <= N {
        i += 1;
        if i > j * (N / D) {
            results[j - 1] = (i - 1) as f32;
            j += 1;
        }
    }
    results
}

fn correct_argmax<const N: usize, const D: usize>(arr: [f32; N]) -> [u32; D] {
    let mut max = 0.0;
    let mut max_index: u32 = 0;
    let mut results: [u32; D] = [0; D];
    let mut i = 0;
    let mut j = 1;
    while i <= N {
        if i >= (j * N / D) {
            results[j - 1] = max_index;
            max = 0.0;
            max_index = 0;
            j += 1;
        }
        if i == N {
            break;
        }
        if arr[i] > max {
            max = arr[i];
            max_index = i as u32;
        }
        i += 1;
    }
    results
}

fn reduce_sum_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut rng());
    }
    let results = run_reduce(&v, N, D, "fast_sum_f32");
    assert_eq!(approx(results, 4), correct_sum::<N, D>());
}

fn reduce_max_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut rng());
    }
    let results = run_reduce(&v, N, D, "fast_max_f32");
    assert_eq!(approx(results, 4), correct_max::<N, D>());
}

fn reduce_argmax_case<const N: usize, const D: usize>() {
    let mut v = create_array::<N>();
    if D == 1 {
        // Hardens 1-dimensional test cases
        v.shuffle(&mut rng());
    }
    let results: Vec<u32> = run_reduce(&v, N, D, "fast_argmax_f32");
    assert_eq!(results, correct_argmax::<N, D>(v));
}

#[test]
fn reduce_sum1() {
    reduce_sum_case::<9, 1>();
    reduce_sum_case::<6, 1>();
    reduce_sum_case::<10, 1>();
    reduce_sum_case::<64, 1>();
    reduce_sum_case::<128, 1>();
    reduce_sum_case::<256, 1>();
    reduce_sum_case::<512, 1>();
    reduce_sum_case::<1024, 1>();
    reduce_sum_case::<2048, 1>();
    reduce_sum_case::<4096, 1>();
}

#[test]
fn reduce_sum2() {
    reduce_sum_case::<6, 2>();
    reduce_sum_case::<10, 2>();
    reduce_sum_case::<64, 2>();
    reduce_sum_case::<128, 2>();
    reduce_sum_case::<256, 2>();
    reduce_sum_case::<512, 2>();
    reduce_sum_case::<1024, 2>();
    reduce_sum_case::<2048, 2>();
    reduce_sum_case::<4096, 2>();
}

#[test]
fn reduce_max() {
    reduce_max_case::<6, 1>();
    reduce_max_case::<9, 1>();
    reduce_max_case::<10, 1>();
    reduce_max_case::<64, 1>();
    reduce_max_case::<128, 1>();
    reduce_max_case::<256, 1>();
    reduce_max_case::<512, 1>();
    reduce_max_case::<1024, 1>();
    reduce_max_case::<2048, 1>();
    reduce_max_case::<4096, 1>();

    reduce_max_case::<6, 2>();
    reduce_max_case::<10, 2>();
    reduce_max_case::<64, 2>();
    reduce_max_case::<128, 2>();
    reduce_max_case::<256, 2>();
    reduce_max_case::<512, 2>();
    reduce_max_case::<1024, 2>();
    reduce_max_case::<2048, 2>();
    reduce_max_case::<4096, 2>();

    reduce_max_case::<6, 3>();
    reduce_max_case::<10, 3>();
    reduce_max_case::<64, 3>();
    reduce_max_case::<128, 3>();
    reduce_max_case::<256, 3>();
    reduce_max_case::<512, 3>();
    reduce_max_case::<1024, 3>();
    reduce_max_case::<2048, 3>();
    reduce_max_case::<4096, 3>();
}

#[test]
fn reduce_argmax() {
    reduce_argmax_case::<6, 1>();
    reduce_argmax_case::<9, 1>();
    reduce_argmax_case::<10, 1>();
    reduce_argmax_case::<64, 1>();
    reduce_argmax_case::<128, 1>();
    reduce_argmax_case::<256, 1>();
    reduce_argmax_case::<512, 1>();
    reduce_argmax_case::<1024, 1>();
    reduce_argmax_case::<2048, 1>();
}

#[test]
fn reduce_argmax2() {
    reduce_argmax_case::<6, 2>();
    reduce_argmax_case::<10, 2>();
    reduce_argmax_case::<64, 2>();
    reduce_argmax_case::<128, 2>();
    reduce_argmax_case::<256, 2>();
    reduce_argmax_case::<512, 2>();
    reduce_argmax_case::<1024, 2>();
    reduce_argmax_case::<2048, 2>();
    reduce_argmax_case::<4096, 2>();
}

#[test]
fn softmax() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let last_dim = 4096;
    let n = 200;
    let mut v = vec![0.0; n * last_dim];
    for i in 0..n {
        v[i * last_dim] = 20.0;
    }
    let results = run_softmax(&v, last_dim, "softmax_f32");
    let results = approx(results, 4);
    assert_eq!(
        results.iter().map(|&s| s.round() as usize).sum::<usize>(),
        n
    );
    assert_eq!(results[0], 1.0);
    assert_eq!(results[1], 0.0);
    assert_eq!(results[last_dim], 1.0);
    assert_eq!(results[2 * last_dim], 1.0);

    let v = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 3;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0900, 0.2447, 0.6652, 0.0900, 0.2447, 0.6652]
    );

    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f16");
    assert_eq!(
        approx_f16(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2332, 0.6338]
    );

    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_bf16");
    assert_eq!(
        approx_bf16(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0859, 0.2324, 0.6328]
    );
}

#[allow(clippy::too_many_arguments)]
fn run_where_cond<I: Clone, T: Clone>(
    shape: &[usize],
    cond: &[I],
    (cond_stride, cond_offset): (Vec<usize>, usize),
    left_true: &[T],
    (left_stride, left_offset): (Vec<usize>, usize),
    right_false: &[T],
    (_right_stride, _right_offset): (Vec<usize>, usize),
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let length = cond.len();
    let cond = device
        .new_buffer_with_data(
            cond.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(cond),
            options,
        )
        .unwrap();
    let left = device
        .new_buffer_with_data(
            left_true.as_ptr() as *const core::ffi::c_void,
            length * core::mem::size_of::<T>(),
            options,
        )
        .unwrap();
    let right = device
        .new_buffer_with_data(
            right_false.as_ptr() as *const core::ffi::c_void,
            length * core::mem::size_of::<T>(),
            options,
        )
        .unwrap();

    let output = device
        .new_buffer(length * core::mem::size_of::<T>(), options)
        .unwrap();
    let cond = BufferOffset {
        buffer: &cond,
        offset_in_bytes: cond_offset,
    };
    let left = BufferOffset {
        buffer: &left,
        offset_in_bytes: left_offset,
    };
    let right = BufferOffset {
        buffer: &right,
        offset_in_bytes: cond_offset,
    };
    call_where_cond(
        &device,
        &encoder,
        &kernels,
        name,
        size_of::<T>(),
        shape,
        cond,
        &cond_stride,
        true,
        left,
        &left_stride,
        true,
        right,
        &cond_stride,
        true,
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, length)
}

#[test]
fn where_cond() {
    let shape = vec![6];
    let cond = vec![0u8, 1, 0, 0, 1, 1];
    let cond_l = (vec![1], 0);
    let left_true = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let left_l = (vec![1], 0);
    let right_false = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let right_l = (vec![1], 0);
    let results = run_where_cond(
        &shape,
        &cond,
        cond_l,
        &left_true,
        left_l,
        &right_false,
        right_l,
        "where_u8_f32",
    );
    assert_eq!(approx(results, 4), vec![-1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0]);
}
#[test]
fn where_cond_u32_f32() {
    let shape = vec![6];
    let cond = vec![0u32, 1, 0, 0, 1, 1];
    let cond_l = (vec![1], 0);
    let left_true = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let left_l = (vec![1], 0);
    let right_false = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let right_l = (vec![1], 0);
    let results = run_where_cond(
        &shape,
        &cond,
        cond_l,
        &left_true,
        left_l,
        &right_false,
        right_l,
        "where_u32_f32",
    );
    assert_eq!(approx(results, 4), vec![-1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0]);
}

#[allow(clippy::too_many_arguments)]
fn run_mlx_gemm<T: Clone>(
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[T],
    lhs_stride: &[usize],
    lhs_offset: usize,
    rhs: &[T],
    rhs_stride: &[usize],
    rhs_offset: usize,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;

    let lhs = device
        .new_buffer_with_data(
            lhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(lhs),
            options,
        )
        .unwrap();
    let rhs = device
        .new_buffer_with_data(
            rhs.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(rhs),
            options,
        )
        .unwrap();
    let length = b * m * n;
    let output = device
        .new_buffer(length * core::mem::size_of::<T>(), options)
        .unwrap();
    call_mlx_gemm(
        &device,
        &encoder,
        &kernels,
        dtype,
        (b, m, n, k),
        lhs_stride,
        lhs_offset,
        &lhs,
        rhs_stride,
        rhs_offset,
        &rhs,
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, length)
}

#[test]
fn mlx_gemm() {
    let (b, m, n, k) = (1, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    let results = run_mlx_gemm(
        GemmDType::F32,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        0,
    );
    assert_eq!(
        approx(results, 4),
        vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
    );

    let (b, m, n, k) = (2, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    let results = run_mlx_gemm(
        GemmDType::F32,
        (b, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        0,
    );
    assert_eq!(
        approx(results, 4),
        vec![
            20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 344.0, 365.0, 386.0, 407.0, 488.0,
            518.0, 548.0, 578.0
        ]
    );

    // OFFSET
    let (b, m, n, k) = (2, 2, 4, 3);
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    // Manually set batch_size=1 and offset 12 elements * 4 the number of bytes for f32
    let results = run_mlx_gemm(
        GemmDType::F32,
        (1, m, n, k),
        &lhs,
        &[m * k, k, 1],
        0,
        &rhs,
        &[n * k, n, 1],
        12 * 4,
    );
    assert_eq!(
        approx(results, 4),
        vec![56.0, 59.0, 62.0, 65.0, 200.0, 212.0, 224.0, 236.0]
    );

    // bgemm sanity test
    {
        let (b, m, n, k) = (1, 2, 4, 3);
        let lhs: Vec<bf16> = (0..b * m * k).map(|f| bf16::from_f32(f as f32)).collect();
        let rhs: Vec<bf16> = (0..b * n * k).map(|f| bf16::from_f32(f as f32)).collect();
        let results = run_mlx_gemm(
            GemmDType::BF16,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            0,
            &rhs,
            &[n * k, n, 1],
            0,
        );
        assert_eq!(
            approx_bf16(results, 4),
            vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
        );
    }

    {
        // hgemm sanity test
        let (b, m, n, k) = (1, 2, 4, 3);
        let lhs: Vec<f16> = (0..b * m * k).map(|f| f16::from_f32(f as f32)).collect();
        let rhs: Vec<f16> = (0..b * n * k).map(|f| f16::from_f32(f as f32)).collect();
        let results = run_mlx_gemm(
            GemmDType::F16,
            (b, m, n, k),
            &lhs,
            &[m * k, k, 1],
            0,
            &rhs,
            &[n * k, n, 1],
            0,
        );
        assert_eq!(
            approx_f16(results, 4),
            vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
        );
    }
}

fn run_random<T: Clone>(name: &'static str, seed: u64, length: usize, a: f32, b: f32) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let options = RESOURCE_OPTIONS;
    let output = device
        .new_buffer(length * core::mem::size_of::<T>(), options)
        .unwrap();

    let seed = device
        .new_buffer_with_data(
            &seed as *const u64 as *const core::ffi::c_void,
            std::mem::size_of::<u64>(),
            options,
        )
        .unwrap();

    if name.starts_with("rand_uniform") {
        call_random_uniform(
            &device, &encoder, &kernels, name, a, b, length, &seed, &output,
        )
        .unwrap();
    } else {
        call_random_normal(
            &device, &encoder, &kernels, name, a, b, length, &seed, &output,
        )
        .unwrap();
    }
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, length)
}

#[test]
fn random() {
    fn calc_mean(data: &[f32]) -> f32 {
        let sum = data.iter().sum::<f32>();
        let count = data.len();
        assert!(count > 0);
        sum / count as f32
    }

    fn calc_stddev(data: &[f32]) -> f32 {
        let mean = calc_mean(data);
        let count = data.len();
        assert!(count > 0);

        let variance = data
            .iter()
            .map(|value| {
                let diff = mean - *value;
                diff * diff
            })
            .sum::<f32>()
            / count as f32;

        variance.sqrt()
    }

    let shape = [1024, 10];

    let length = shape.iter().product::<usize>();
    let seed = 299792458u64;

    let min = -30.0;
    let max = 30.0;
    let mean = 100.0;
    let stddev = 50.0;

    macro_rules! validate_random {
        ($type:ty) => {
            let results: Vec<f32> = run_random::<$type>(
                concat!("rand_uniform_", stringify!($type)),
                seed,
                length,
                min,
                max,
            )
            .into_iter()
            .map(f32::from)
            .collect();
            results.iter().for_each(|v| {
                assert!(*v >= min && *v <= max);
            });
            assert!(calc_mean(&results) > -1.0 && calc_mean(&results) < 1.0);

            let results: Vec<f32> = run_random::<$type>(
                concat!("rand_normal_", stringify!($type)),
                seed,
                length,
                mean,
                stddev,
            )
            .into_iter()
            .map(f32::from)
            .collect();
            assert!((calc_mean(&results) - mean).abs() < mean / 10.0);
            assert!((calc_stddev(&results) - stddev).abs() < stddev / 10.0);
        };
    }

    validate_random!(f32);
    validate_random!(f16);
    validate_random!(bf16);
}

fn run_scatter_add<T: Clone, I: Clone + std::fmt::Debug>(
    input: &[T],
    ids: &[I],
    shape: &[usize],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let options = RESOURCE_OPTIONS;
    let input_buffer = new_buffer(&device, input);
    let ids_buffer = new_buffer(&device, ids);
    let output = device
        .new_buffer(std::mem::size_of_val(input), options)
        .unwrap();
    call_scatter(
        &device,
        &encoder,
        &kernels,
        name,
        shape,
        shape,
        dim,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&ids_buffer),
        BufferOffset::zero_offset(&output),
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, input.len())
}

#[test]
fn scatter_add() {
    let ids_u8 = [0u8, 0, 1, 0, 2, 2, 3, 3];
    let ids_u32 = [0u32, 0, 1, 0, 2, 2, 3, 3];
    let ids_i64 = [0i64, 0, 1, 0, 2, 2, 3, 3];

    let input_f32 = [5.0f32, 1.0, 7.0, 2.0, 3.0, 2.0, 1.0, 3.0];
    let input_f16 = input_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let input_bf16 = input_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    let output_dim1_f32 = vec![8.0, 7.0, 5.0, 4.0, 0.0, 0.0, 0.0, 0.0];
    let output_dim1_f16 = output_dim1_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let output_dim1_bf16 = output_dim1_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    let output_dim2_f32 = vec![5.0, 3.0, 7.0, 0.0, 3.0, 2.0, 1.0, 3.0];
    let output_dim2_f16 = output_dim2_f32
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let output_dim2_bf16 = output_dim2_f32
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();

    for (shape, output_f32, output_f16, output_bf16) in [
        (vec![8], output_dim1_f32, output_dim1_f16, output_dim1_bf16),
        (
            vec![4, 2],
            output_dim2_f32,
            output_dim2_f16,
            output_dim2_bf16,
        ),
    ] {
        for results in [
            run_scatter_add(&input_f32, &ids_u8, &shape, 0, "sa_u8_f32"),
            run_scatter_add(&input_f32, &ids_u32, &shape, 0, "sa_u32_f32"),
            run_scatter_add(&input_f32, &ids_i64, &shape, 0, "sa_i64_f32"),
        ] {
            assert_eq!(results, output_f32);
        }
        for results in [
            run_scatter_add(&input_f16, &ids_u8, &shape, 0, "sa_u8_f16"),
            run_scatter_add(&input_f16, &ids_u32, &shape, 0, "sa_u32_f16"),
            run_scatter_add(&input_f16, &ids_i64, &shape, 0, "sa_i64_f16"),
        ] {
            assert_eq!(results, output_f16);
        }
        for results in [
            run_scatter_add(&input_bf16, &ids_u8, &shape, 0, "sa_u8_bf16"),
            run_scatter_add(&input_bf16, &ids_u32, &shape, 0, "sa_u32_bf16"),
            run_scatter_add(&input_bf16, &ids_i64, &shape, 0, "sa_i64_bf16"),
        ] {
            assert_eq!(results, output_bf16);
        }
    }
}

fn run_index_add<T: Clone, I: Clone + std::fmt::Debug>(
    left: &[T],
    right: &[T],
    indices: &[I],
    shape: &[usize],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let input_buffer = new_buffer(&device, right);
    let output = new_buffer(&device, left);
    let indices_buffer = new_buffer(&device, indices);
    call_index_add(
        &device,
        &encoder,
        &kernels,
        name,
        shape,
        shape,
        shape,
        dim,
        BufferOffset::zero_offset(&input_buffer),
        BufferOffset::zero_offset(&indices_buffer),
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();
    read_to_vec(&output, left.len())
}

#[test]
fn index_add() {
    let left = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let right = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
    let indices = vec![0u32, 1, 0, 1, 0, 1];
    let shape = vec![6];

    // u32, f32
    {
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u32, f16
    {
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u32, bf16
    {
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u32_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, f32
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, f16
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // u8, bf16
    {
        let indices = indices.iter().map(|v| *v as u8).collect::<Vec<_>>();
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_u8_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, f32
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_f32");
        assert_eq!(results, vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, f16
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let left = left.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_f16");
        assert_eq!(approx_f16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // i64, bf16
    {
        let indices = indices.iter().map(|v| *v as i64).collect::<Vec<_>>();
        let left = left.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let right = right.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let results = run_index_add(&left, &right, &indices, &shape, 0, "ia_i64_bf16");
        assert_eq!(approx_bf16(results, 4), vec![4.0, 5.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

fn run_pool2d<T: Clone>(
    v: &[T],
    (w_k, h_k): (usize, usize),
    (w_stride, h_stride): (usize, usize),
    shape: &[usize],
    strides: &[usize],
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();
    let out_w = (shape[2] - w_k) / w_stride + 1;
    let out_h = (shape[3] - h_k) / h_stride + 1;
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, &vec![0.0f32; dst_el]);
    let kernels = Kernels::new();
    call_pool2d(
        &device, &encoder, &kernels, name, shape, strides, out_w, out_h, w_k, h_k, w_stride,
        h_stride, &input, &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, dst_el)
}

#[test]
fn max_pool2d_f32() {
    // kernel 2 stride 1
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f32",
    );
    let expected = vec![5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f32",
    );
    let expected = vec![5.0, 7.0, 13.0, 15.0];
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_f16() {
    // kernel 2 stride 1
    let v: Vec<half::f16> = (0..16).map(|v| half::f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f16",
    );
    let expected = [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0]
        .iter()
        .map(|v| half::f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<half::f16> = (0..16).map(|v| half::f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_f16",
    );
    let expected = [5.0, 7.0, 13.0, 15.0]
        .iter()
        .map(|v| half::f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_bf16() {
    // kernel 2 stride 1
    let v: Vec<half::bf16> = (0..16).map(|v| half::bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_bf16",
    );
    let expected = [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0]
        .iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<half::bf16> = (0..16).map(|v| half::bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_bf16",
    );
    let expected = [5.0, 7.0, 13.0, 15.0]
        .iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_u8() {
    // kernel 2 stride 1
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u8",
    );
    let expected = vec![5, 6, 7, 9, 10, 11, 13, 14, 15];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u8",
    );
    let expected = vec![5, 7, 13, 15];
    assert_eq!(results, expected);
}

#[test]
fn max_pool2d_u32() {
    // kernel 2 stride 1
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u32",
    );
    let expected = vec![5, 6, 7, 9, 10, 11, 13, 14, 15];
    assert_eq!(results, expected);

    // kernel 2 stride 2
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 2;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "max_pool2d_u32",
    );
    let expected = vec![5, 7, 13, 15];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_f32() {
    // kernel 2 stride 1
    let v: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_f32",
    );
    let expected = vec![
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_f16() {
    // kernel 2 stride 1
    let v: Vec<f16> = (0..16).map(|v| f16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_f16",
    );
    let expected = [
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ]
    .iter()
    .map(|v| f16::from_f32(*v))
    .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_bf16() {
    // kernel 2 stride 1
    let v: Vec<bf16> = (0..16).map(|v| bf16::from_f32(v as f32)).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_bf16",
    );
    let expected = [
        2.5000, 3.5000, 4.5000, 6.5000, 7.5000, 8.5000, 10.5000, 11.5000, 12.5000,
    ]
    .iter()
    .map(|v| bf16::from_f32(*v))
    .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_u8() {
    // kernel 2 stride 1
    let v: Vec<u8> = (0..16).map(|v| v as u8).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_u8",
    );
    let expected = vec![2, 3, 4, 6, 7, 8, 10, 11, 12];
    assert_eq!(results, expected);
}

#[test]
fn avg_pool2d_u32() {
    // kernel 2 stride 1
    let v: Vec<u32> = (0..16).map(|v| v as u32).collect();
    let shape = vec![1, 1, 4, 4];
    let strides = vec![16, 16, 4, 1];
    let kernel = 2;
    let stride = 1;
    let results = run_pool2d(
        &v,
        (kernel, kernel),
        (stride, stride),
        &shape,
        &strides,
        "avg_pool2d_u32",
    );
    let expected = vec![2, 3, 4, 6, 7, 8, 10, 11, 12];
    assert_eq!(results, expected);
}

#[allow(clippy::too_many_arguments)]
fn run_conv_transpose1d<T: Clone>(
    input: &[T],
    input_shape: &[usize],
    input_stride: &[usize],
    kernel: &[T],
    kernel_shape: &[usize],
    kernel_stride: &[usize],
    dilation: usize,
    stride: usize,
    padding: usize,
    out_padding: usize,
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let c_out = kernel_shape[1];
    let k_size = kernel_shape[2];
    let b_size = input_shape[0];
    let l_in = input_shape[2];
    let l_out = (l_in - 1) * stride - 2 * padding + dilation * (k_size - 1) + out_padding + 1;
    let dst_el = c_out * l_out * b_size;

    let input = new_buffer(&device, input);
    let kernel = new_buffer(&device, kernel);
    let output = new_buffer(&device, &vec![0.0f32; dst_el]);
    let kernels = Kernels::new();

    call_conv_transpose1d(
        &device,
        &encoder,
        &kernels,
        name,
        dilation,
        stride,
        padding,
        out_padding,
        c_out,
        l_out,
        b_size,
        input_shape,
        input_stride,
        kernel_shape,
        kernel_stride,
        &input,
        0,
        &kernel,
        0,
        &output,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    read_to_vec(&output, dst_el)
}

#[test]
fn conv_transpose1d_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel = vec![1.0f32, 2.0, 3.0, 4.0];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_f32",
    );

    let expected = vec![1., 4., 10., 20., 25., 24., 16.];
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_f16() {
    let input: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_f16",
    );

    let expected = [1., 4., 10., 20., 25., 24., 16.]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_bf16() {
    let input: Vec<bf16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<bf16> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_bf16",
    );

    let expected = [1., 4., 10., 20., 25., 24., 16.]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_u8() {
    let input: Vec<u8> = vec![1, 2, 3, 4];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<u8> = vec![1, 2, 3, 4];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_u8",
    );

    let expected = vec![1, 4, 10, 20, 25, 24, 16];
    assert_eq!(results, expected);
}

#[test]
fn conv_transpose1d_u32() {
    let input: Vec<u32> = vec![1, 2, 3, 4];
    let input_shape = &[1, 1, 4];
    let input_stride = &[4, 4, 1];

    let kernel: Vec<u32> = vec![1, 2, 3, 4];
    let kernel_shape = &[1, 1, 4];
    let kernel_stride = &[4, 4, 1];

    let results = run_conv_transpose1d(
        &input,
        input_shape,
        input_stride,
        &kernel,
        kernel_shape,
        kernel_stride,
        1,
        1,
        0,
        0,
        "conv_transpose1d_u32",
    );

    let expected = vec![1, 4, 10, 20, 25, 24, 16];
    assert_eq!(results, expected);
}

#[test]
fn const_fill() {
    fn constant_fill<T: Clone + EncoderParam>(name: &'static str, len: usize, value: T) -> Vec<T> {
        let dev = device();
        let kernels = Kernels::new();
        let commands = commands(&dev);
        let encoder = commands.command_encoder().unwrap();
        let buffer = dev
            .new_buffer(len * std::mem::size_of::<T>(), RESOURCE_OPTIONS)
            .unwrap();
        call_const_fill(&dev, &encoder, &kernels, name, len, &buffer, value).unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();
        read_to_vec::<T>(&buffer, len)
    }
    fn test<T: Clone + Copy + EncoderParam + PartialEq + std::fmt::Debug, F: FnOnce(f32) -> T>(
        name: &'static str,
        f: F,
    ) {
        let len = rand::rng().random_range(2..16) * rand::rng().random_range(4..16);
        let value = rand::rng().random_range(1. ..19.);
        let value = f(value);
        let v = constant_fill::<T>(name, len, value);
        assert_eq!(v, vec![value; len])
    }
    test::<u8, _>("fill_u8", |v| v as u8);
    test::<u32, _>("fill_u32", |v| v as u32);
    test::<i64, _>("fill_i64", |v| v as i64);
    test::<f16, _>("fill_f16", f16::from_f32);
    test::<bf16, _>("fill_bf16", bf16::from_f32);
    test::<f32, _>("fill_f32", |v| v);
}

#[test]
fn commands_creation_and_encoder() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    let commands = Commands::new(queue, &residency_set).unwrap();

    let encoder = commands.command_encoder().unwrap();
    drop(encoder);
}

#[test]
fn commands_concurrent_acquisition() {
    std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", "2");

    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(&device));
    let commands = Arc::new(Commands::new(queue, &residency_set).unwrap());

    let mut handles = vec![];

    for _ in 0..16 {
        let c = Arc::clone(&commands);
        handles.push(thread::spawn(move || {
            let encoder = c.command_encoder().unwrap();
            drop(encoder);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    commands.wait_until_completed().unwrap();
}

#[test]
fn residency_set_batch_insert_remove() {
    use objc2_metal::MTLResidencySet;

    let device = device();
    let set = ResidencySet::new(&device);
    let Some(raw) = set.raw() else {
        // Residency sets are unsupported on this device/OS; the set no-ops.
        return;
    };

    let bufs: Vec<Buffer> = (0..3).map(|i| new_buffer(&device, &[i as f32])).collect();
    let base = raw.allocationCount();

    set.insert_batch(&bufs);
    assert_eq!(raw.allocationCount(), base + bufs.len());
    set.remove_batch(&bufs);
    assert_eq!(raw.allocationCount(), base);

    // Empty batches are valid and leave the set untouched.
    set.insert_batch(std::iter::empty());
    set.remove_batch(std::iter::empty());
    assert_eq!(raw.allocationCount(), base);
}

// Fused gated-DeltaNet decode step: confirms the kernel compiles and
// loads as a real Metal pipeline before any numeric-correctness work is
// asked to rely on it -- same de-risk-spike-before-wiring precedent as
// kernel_mul_mv_id_pipelines_load.
#[test]
fn kernel_gdn_decode_step_pipeline_loads() {
    let device = device();
    let kernels = Kernels::new();
    kernels
        .load_pipeline(&device, Source::Gdn, "kernel_gdn_decode_step_f32")
        .unwrap_or_else(|e| {
            panic!("kernel_gdn_decode_step_f32 should load as a Metal compute pipeline: {e}")
        });
}

// Scalar Rust reference for the fused kernel's math (see gdn.metal's own
// doc comment for the derivation):
//   s_dec[i][j] = g * s_in[i][j]
//   kv_mem[j]   = sum_i s_dec[i][j] * k[i]
//   delta[j]    = (v[j] - kv_mem[j]) * beta
//   s_out[i][j] = s_dec[i][j] + k[i] * delta[j]
//   out[j]      = sum_i s_out[i][j] * q[i]
// Per (batch, head) -- q/k: [hk], v: [hv], g_log/beta: scalar, state: [hk, hv].
// `g_log` is the raw decay-gate log, NOT pre-exponentiated -- matches the
// kernel's own convention (it exponentiates internally).
#[allow(clippy::too_many_arguments)]
fn gdn_decode_step_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_log: f32,
    beta: f32,
    state_in: &[f32],
    hk: usize,
    hv: usize,
) -> (Vec<f32>, Vec<f32>) {
    let g = g_log.exp();
    let mut state_out = vec![0f32; hk * hv];
    let mut out = vec![0f32; hv];
    for j in 0..hv {
        let mut kv_mem = 0f32;
        for i in 0..hk {
            kv_mem += (g * state_in[i * hv + j]) * k[i];
        }
        let delta = (v[j] - kv_mem) * beta;
        let mut acc = 0f32;
        for i in 0..hk {
            let s_new = g * state_in[i * hv + j] + k[i] * delta;
            state_out[i * hv + j] = s_new;
            acc += s_new * q[i];
        }
        out[j] = acc;
    }
    (out, state_out)
}

// Runs the real kernel on random inputs at the given shape and compares
// against the scalar reference above, per (batch, head). `hv` deliberately
// not always a multiple of the threadgroup width (min(hv, 64) in
// call_gdn_decode_step_f32) to exercise the kernel's own `j >= args.hv`
// bounds check, not just the common case.
fn run_gdn_decode_step_and_check(b: usize, h: usize, hk: usize, hv: usize) {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let mut rng = rng();
    fn randf(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| (rng.random::<f32>() - 0.5) * scale)
            .collect()
    }
    let q = randf(&mut rng, b * h * hk, 0.2);
    let k = randf(&mut rng, b * h * hk, 0.2);
    let v = randf(&mut rng, b * h * hv, 2.0);
    // Sample the actual decay gate in (0,1) (a realistic range, including
    // the strong-decay end near 0), then take its log -- the kernel and
    // the reference both take g_log directly now, not the pre-exp'd gate.
    let g_decay_vals: Vec<f32> = (0..b * h)
        .map(|_| rng.random::<f32>() * 0.9 + 0.05)
        .collect();
    let g_log_vals: Vec<f32> = g_decay_vals.iter().map(|g| g.ln()).collect();
    let beta_vals: Vec<f32> = (0..b * h)
        .map(|_| rng.random::<f32>() * 0.9 + 0.05)
        .collect();
    let state_in = randf(&mut rng, b * h * hk * hv, 1.0);

    let q_buf = new_buffer(&device, &q);
    let k_buf = new_buffer(&device, &k);
    let v_buf = new_buffer(&device, &v);
    let g_buf = new_buffer(&device, &g_log_vals);
    let beta_buf = new_buffer(&device, &beta_vals);
    let state_in_buf = new_buffer(&device, &state_in);
    let state_out_buf = device
        .new_buffer(
            b * h * hk * hv * std::mem::size_of::<f32>(),
            RESOURCE_OPTIONS,
        )
        .unwrap();
    let out_buf = device
        .new_buffer(b * h * hv * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    call_gdn_decode_step_f32(
        &device,
        &encoder,
        &kernels,
        b,
        h,
        hk,
        hv,
        &BufferOffset::zero_offset(&q_buf),
        &BufferOffset::zero_offset(&k_buf),
        &BufferOffset::zero_offset(&v_buf),
        &BufferOffset::zero_offset(&g_buf),
        &BufferOffset::zero_offset(&beta_buf),
        &BufferOffset::zero_offset(&state_in_buf),
        &state_out_buf,
        &out_buf,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let got_out: Vec<f32> = read_to_vec(&out_buf, b * h * hv);
    let got_state: Vec<f32> = read_to_vec(&state_out_buf, b * h * hk * hv);

    let mut max_out_diff = 0f32;
    let mut max_state_diff = 0f32;
    for bh in 0..b * h {
        let (expected_out, expected_state) = gdn_decode_step_reference(
            &q[bh * hk..(bh + 1) * hk],
            &k[bh * hk..(bh + 1) * hk],
            &v[bh * hv..(bh + 1) * hv],
            g_log_vals[bh],
            beta_vals[bh],
            &state_in[bh * hk * hv..(bh + 1) * hk * hv],
            hk,
            hv,
        );
        for j in 0..hv {
            let diff = (got_out[bh * hv + j] - expected_out[j]).abs();
            max_out_diff = max_out_diff.max(diff);
        }
        for e in 0..hk * hv {
            let diff = (got_state[bh * hk * hv + e] - expected_state[e]).abs();
            max_state_diff = max_state_diff.max(diff);
        }
    }
    println!(
        "gdn_decode_step b={b} h={h} hk={hk} hv={hv}: out_diff={max_out_diff:.8} state_diff={max_state_diff:.8}"
    );
    assert!(
        max_out_diff < 5e-5,
        "b={b} h={h} hk={hk} hv={hv}: output mismatch, max diff = {max_out_diff}"
    );
    assert!(
        max_state_diff < 5e-5,
        "b={b} h={h} hk={hk} hv={hv}: state mismatch, max diff = {max_state_diff}"
    );
}

#[test]
fn kernel_gdn_decode_step_matches_scalar_reference_tiny() {
    // Small, odd shapes -- hv=5 is not a multiple of min(hv,64)'s
    // threadgroup width, exercising the bounds check on every thread group
    // boundary, not just the last one.
    run_gdn_decode_step_and_check(1, 2, 4, 4);
    run_gdn_decode_step_and_check(1, 3, 7, 5);
    run_gdn_decode_step_and_check(2, 2, 6, 6);
}

#[test]
fn kernel_gdn_decode_step_matches_scalar_reference_production_shape() {
    // b=1, h=32, hk=128, hv=128 -- a real production shape from a
    // Qwen3.5-family hybrid gated-DeltaNet model.
    run_gdn_decode_step_and_check(1, 32, 128, 128);
}

// Regression test for a real bug found while integrating this kernel: a
// caller's `v` was a `narrow()`'d slice of a shared QKV-split buffer with
// a genuine nonzero byte offset, which an earlier version of
// call_gdn_decode_step_f32 (bare &Buffer params, no offset) silently
// ignored -- reading the wrong region of the buffer. This test reproduces
// that shape directly: q/k/v/g/beta/state_in are all slices into ONE
// larger packed buffer at different offsets (not six independent
// zero-offset allocations, which is what every other test here uses and
// exactly why this bug slipped past them), so a wrong or ignored offset
// reads cross-contaminated data and fails the numeric check hard, not
// subtly.
#[test]
fn kernel_gdn_decode_step_respects_nonzero_buffer_offsets() {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let (b, h, hk, hv) = (1usize, 4usize, 8usize, 8usize);
    let mut rng = rng();
    fn randf(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| (rng.random::<f32>() - 0.5) * scale)
            .collect()
    }

    // Pack q, k, v, g, beta, state_in back-to-back into one buffer, each
    // preceded by a "decoy" region of the *next* tensor's own values --
    // i.e. deliberately construct it so that reading from byte offset 0
    // instead of the real offset would read a DIFFERENT tensor's data,
    // not just garbage. Layout: [decoy_q_sized][q][k][v][g][beta][state_in].
    let q = randf(&mut rng, b * h * hk, 0.2);
    let k = randf(&mut rng, b * h * hk, 0.2);
    let v = randf(&mut rng, b * h * hv, 2.0);
    // Values don't need to be a realistic post-exp decay-gate range here
    // (unlike run_gdn_decode_step_and_check's own g) -- this test is
    // about offset correctness, not numeric range, and the kernel
    // exponentiates internally regardless.
    let g_log_vals: Vec<f32> = (0..b * h)
        .map(|_| rng.random::<f32>() * 0.9 + 0.05)
        .collect();
    let beta_vals: Vec<f32> = (0..b * h)
        .map(|_| rng.random::<f32>() * 0.9 + 0.05)
        .collect();
    let state_in = randf(&mut rng, b * h * hk * hv, 1.0);
    let decoy = randf(&mut rng, b * h * hk, 999.0); // same size as q, deliberately huge-magnitude so a wrong-offset read is obviously wrong

    let f32_size = std::mem::size_of::<f32>();
    let packed: Vec<f32> = [
        &decoy[..],
        &q[..],
        &k[..],
        &v[..],
        &g_log_vals[..],
        &beta_vals[..],
        &state_in[..],
    ]
    .concat();
    let packed_buf = new_buffer(&device, &packed);

    let mut offset_elems = decoy.len();
    let q_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };
    offset_elems += q.len();
    let k_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };
    offset_elems += k.len();
    let v_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };
    offset_elems += v.len();
    let g_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };
    offset_elems += g_log_vals.len();
    let beta_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };
    offset_elems += beta_vals.len();
    let state_in_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };

    let state_out_buf = device
        .new_buffer(b * h * hk * hv * f32_size, RESOURCE_OPTIONS)
        .unwrap();
    let out_buf = device
        .new_buffer(b * h * hv * f32_size, RESOURCE_OPTIONS)
        .unwrap();

    call_gdn_decode_step_f32(
        &device,
        &encoder,
        &kernels,
        b,
        h,
        hk,
        hv,
        &q_off,
        &k_off,
        &v_off,
        &g_off,
        &beta_off,
        &state_in_off,
        &state_out_buf,
        &out_buf,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let got_out: Vec<f32> = read_to_vec(&out_buf, b * h * hv);
    let got_state: Vec<f32> = read_to_vec(&state_out_buf, b * h * hk * hv);

    let mut max_out_diff = 0f32;
    let mut max_state_diff = 0f32;
    for bh in 0..b * h {
        let (expected_out, expected_state) = gdn_decode_step_reference(
            &q[bh * hk..(bh + 1) * hk],
            &k[bh * hk..(bh + 1) * hk],
            &v[bh * hv..(bh + 1) * hv],
            g_log_vals[bh],
            beta_vals[bh],
            &state_in[bh * hk * hv..(bh + 1) * hk * hv],
            hk,
            hv,
        );
        for j in 0..hv {
            max_out_diff = max_out_diff.max((got_out[bh * hv + j] - expected_out[j]).abs());
        }
        for e in 0..hk * hv {
            max_state_diff =
                max_state_diff.max((got_state[bh * hk * hv + e] - expected_state[e]).abs());
        }
    }
    println!("gdn_decode_step_nonzero_offsets: out_diff={max_out_diff:.8} state_diff={max_state_diff:.8}");
    assert!(
        max_out_diff < 5e-5,
        "nonzero-offset output mismatch, max diff = {max_out_diff}"
    );
    assert!(
        max_state_diff < 5e-5,
        "nonzero-offset state mismatch, max diff = {max_state_diff}"
    );
}
// Fused causal depthwise conv1d + silu: same de-risk-spike-before-wiring
// precedent as the decode-step kernel above.
#[test]
fn kernel_gdn_causal_conv1d_pipelines_load() {
    let device = device();
    let kernels = Kernels::new();
    for name in [
        "kernel_gdn_causal_conv1d_output_f32",
        "kernel_gdn_causal_conv1d_state_f32",
    ] {
        kernels
            .load_pipeline(&device, Source::Gdn, name)
            .unwrap_or_else(|e| panic!("{name} should load as a Metal compute pipeline: {e}"));
    }
}

/// Scalar Rust reference for the fused conv1d kernels' math (see
/// gdn.metal's own doc comment for the derivation): both kernels operate
/// conceptually on `padded = history ++ x` (concat along time, length
/// `hist_len + seq_len`) without ever materializing it.
///   out[t][c]       = silu( sum_k padded[t+k][c] * weight[c][k] ),   0 <= t < seq_len
///   new_state[s][c] = padded[seq_len + s][c],                       0 <= s < hist_len
/// Operates on one batch row at a time (caller loops over `b`).
fn gdn_causal_conv1d_reference(
    x: &[f32],
    history: &[f32],
    weight: &[f32],
    seq_len: usize,
    hist_len: usize,
    channels: usize,
    kernel_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let padded_at = |idx: usize, c: usize| -> f32 {
        if idx < hist_len {
            history[idx * channels + c]
        } else {
            x[(idx - hist_len) * channels + c]
        }
    };
    let mut out = vec![0f32; seq_len * channels];
    for t in 0..seq_len {
        for c in 0..channels {
            let mut acc = 0f32;
            for k in 0..kernel_size {
                acc += padded_at(t + k, c) * weight[c * kernel_size + k];
            }
            out[t * channels + c] = acc / (1.0 + (-acc).exp()); // silu(x) = x * sigmoid(x)
        }
    }
    let mut new_state = vec![0f32; hist_len * channels];
    for s in 0..hist_len {
        for c in 0..channels {
            new_state[s * channels + c] = padded_at(seq_len + s, c);
        }
    }
    (out, new_state)
}

/// Runs both real kernels on random inputs at the given shape and compares
/// against the scalar reference above, per batch row. `channels`
/// deliberately not always a multiple of the threadgroup width
/// (min(channels, 64)) to exercise the kernels' own bounds checks.
fn run_gdn_causal_conv1d_and_check(
    b: usize,
    seq_len: usize,
    hist_len: usize,
    channels: usize,
    kernel_size: usize,
) {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let mut rng = rng();
    fn randf(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| (rng.random::<f32>() - 0.5) * scale)
            .collect()
    }
    let x = randf(&mut rng, b * seq_len * channels, 1.0);
    let history = randf(&mut rng, b * hist_len * channels, 1.0);
    let weight = randf(&mut rng, channels * kernel_size, 0.5);

    let x_buf = new_buffer(&device, &x);
    // `new_buffer_with_data` can't allocate a genuinely zero-size buffer
    // (hist_len == 0 is a real, tested case below) -- pad with one unread
    // dummy element; the kernel's own `idx < hist_len` check never reads
    // `history` at all when `hist_len == 0`, so its content is irrelevant.
    let history_buf = new_buffer(
        &device,
        if history.is_empty() {
            &[0f32]
        } else {
            &history
        },
    );
    let weight_buf = new_buffer(&device, &weight);
    let out_buf = device
        .new_buffer(
            b * seq_len * channels * std::mem::size_of::<f32>(),
            RESOURCE_OPTIONS,
        )
        .unwrap();
    let new_state_buf = device
        .new_buffer(
            (b * hist_len * channels).max(1) * std::mem::size_of::<f32>(),
            RESOURCE_OPTIONS,
        )
        .unwrap();

    call_gdn_causal_conv1d_output_f32(
        &device,
        &encoder,
        &kernels,
        b,
        seq_len,
        hist_len,
        channels,
        kernel_size,
        &BufferOffset::zero_offset(&x_buf),
        &BufferOffset::zero_offset(&history_buf),
        &BufferOffset::zero_offset(&weight_buf),
        &out_buf,
    )
    .unwrap();
    if hist_len > 0 {
        call_gdn_causal_conv1d_state_f32(
            &device,
            &encoder,
            &kernels,
            b,
            seq_len,
            hist_len,
            channels,
            &BufferOffset::zero_offset(&x_buf),
            &BufferOffset::zero_offset(&history_buf),
            &new_state_buf,
        )
        .unwrap();
    }
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let got_out: Vec<f32> = read_to_vec(&out_buf, b * seq_len * channels);
    let got_state: Vec<f32> = if hist_len > 0 {
        read_to_vec(&new_state_buf, b * hist_len * channels)
    } else {
        vec![]
    };

    let mut max_out_diff = 0f32;
    let mut max_state_diff = 0f32;
    for row in 0..b {
        let (expected_out, expected_state) = gdn_causal_conv1d_reference(
            &x[row * seq_len * channels..(row + 1) * seq_len * channels],
            &history[row * hist_len * channels..(row + 1) * hist_len * channels],
            &weight,
            seq_len,
            hist_len,
            channels,
            kernel_size,
        );
        for i in 0..seq_len * channels {
            max_out_diff =
                max_out_diff.max((got_out[row * seq_len * channels + i] - expected_out[i]).abs());
        }
        for i in 0..hist_len * channels {
            max_state_diff = max_state_diff
                .max((got_state[row * hist_len * channels + i] - expected_state[i]).abs());
        }
    }
    println!(
        "gdn_causal_conv1d b={b} seq_len={seq_len} hist_len={hist_len} channels={channels} \
         kernel_size={kernel_size}: out_diff={max_out_diff:.8} state_diff={max_state_diff:.8}"
    );
    assert!(
        max_out_diff < 5e-5,
        "b={b} seq_len={seq_len} hist_len={hist_len}: output mismatch, max diff = {max_out_diff}"
    );
    assert!(
        max_state_diff < 5e-5,
        "b={b} seq_len={seq_len} hist_len={hist_len}: state mismatch, max diff = {max_state_diff}"
    );
}

#[test]
fn kernel_gdn_causal_conv1d_matches_scalar_reference_every_short_length() {
    // Covers the whole native-MTP verify-window range (1-8) at a realistic
    // kernel_size=4 (hist_len=3) -- including the seq_len < hist_len case
    // (seq_len 1, 2), which is the COMMON case at real decode/short-verify
    // shapes, not a rare edge case (found during design: hist_len=3 means
    // seq_len=1 and seq_len=2 both hit the "mixed history+input" branch of
    // the new-state computation).
    for seq_len in 1..=8 {
        run_gdn_causal_conv1d_and_check(2, seq_len, 3, 6, 4);
    }
}

#[test]
fn kernel_gdn_causal_conv1d_matches_scalar_reference_production_shape() {
    // b=1, channels=1536 (confirmed against the cached qwen36-35b-a3b
    // GGUF's real mixed-qkv width: group_count*state_size*2 + inner_size),
    // kernel_size=4 (hist_len=3), at both a decode (seq_len=1) and a
    // verify-window (seq_len=2) shape.
    run_gdn_causal_conv1d_and_check(1, 1, 3, 1536, 4);
    run_gdn_causal_conv1d_and_check(1, 2, 3, 1536, 4);
}

#[test]
fn kernel_gdn_causal_conv1d_handles_zero_history_length() {
    // kernel_size=1 (hist_len=0) is a degenerate but real edge case: no
    // causal history needed at all, output kernel must never read `history`
    // out of bounds, and the state kernel must be skippable (a zero-size
    // grid dimension) without dispatching at all.
    run_gdn_causal_conv1d_and_check(1, 3, 0, 8, 1);
}

// Regression-shaped test for the same offset bug class the decode-step
// kernel found live (kernel_gdn_decode_step_respects_nonzero_buffer_offsets
// above): packs x/history/weight back-to-back into ONE buffer with a decoy
// region ahead of each, at real production-realistic offsets, so a wrong or
// ignored offset reads cross-contaminated data and fails hard.
#[test]
fn kernel_gdn_causal_conv1d_respects_nonzero_buffer_offsets() {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let (b, seq_len, hist_len, channels, kernel_size) = (1usize, 2usize, 3usize, 8usize, 4usize);
    let mut rng = rng();
    fn randf(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| (rng.random::<f32>() - 0.5) * scale)
            .collect()
    }

    let x = randf(&mut rng, b * seq_len * channels, 1.0);
    let history = randf(&mut rng, b * hist_len * channels, 1.0);
    let weight = randf(&mut rng, channels * kernel_size, 0.5);
    let decoy = randf(&mut rng, b * seq_len * channels, 999.0);

    let f32_size = std::mem::size_of::<f32>();
    let packed: Vec<f32> = [&decoy[..], &x[..], &history[..], &weight[..]].concat();
    let packed_buf = new_buffer(&device, &packed);

    let mut offset_elems = decoy.len();
    let x_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };
    offset_elems += x.len();
    let history_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };
    offset_elems += history.len();
    let weight_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };

    let out_buf = device
        .new_buffer(b * seq_len * channels * f32_size, RESOURCE_OPTIONS)
        .unwrap();
    let new_state_buf = device
        .new_buffer(b * hist_len * channels * f32_size, RESOURCE_OPTIONS)
        .unwrap();

    call_gdn_causal_conv1d_output_f32(
        &device,
        &encoder,
        &kernels,
        b,
        seq_len,
        hist_len,
        channels,
        kernel_size,
        &x_off,
        &history_off,
        &weight_off,
        &out_buf,
    )
    .unwrap();
    call_gdn_causal_conv1d_state_f32(
        &device,
        &encoder,
        &kernels,
        b,
        seq_len,
        hist_len,
        channels,
        &x_off,
        &history_off,
        &new_state_buf,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let got_out: Vec<f32> = read_to_vec(&out_buf, b * seq_len * channels);
    let got_state: Vec<f32> = read_to_vec(&new_state_buf, b * hist_len * channels);
    let (expected_out, expected_state) = gdn_causal_conv1d_reference(
        &x,
        &history,
        &weight,
        seq_len,
        hist_len,
        channels,
        kernel_size,
    );

    let mut max_out_diff = 0f32;
    let mut max_state_diff = 0f32;
    for i in 0..seq_len * channels {
        max_out_diff = max_out_diff.max((got_out[i] - expected_out[i]).abs());
    }
    for i in 0..hist_len * channels {
        max_state_diff = max_state_diff.max((got_state[i] - expected_state[i]).abs());
    }
    println!("gdn_causal_conv1d_nonzero_offsets: out_diff={max_out_diff:.8} state_diff={max_state_diff:.8}");
    assert!(
        max_out_diff < 5e-5,
        "nonzero-offset output mismatch, max diff = {max_out_diff}"
    );
    assert!(
        max_state_diff < 5e-5,
        "nonzero-offset state mismatch, max diff = {max_state_diff}"
    );
}

// Fused elementwise gating-tail kernels: same de-risk-spike-before-wiring
// precedent as the kernels above.
#[test]
fn kernel_gdn_preprocessing_gating_pipelines_load() {
    let device = device();
    let kernels = Kernels::new();
    for name in [
        "kernel_gdn_l2_normalize_scale_f32",
        "kernel_gdn_decay_beta_gate_f32",
    ] {
        kernels
            .load_pipeline(&device, Source::Gdn, name)
            .unwrap_or_else(|e| panic!("{name} should load as a Metal compute pipeline: {e}"));
    }
}

/// Scalar Rust reference for `kernel_gdn_l2_normalize_scale_f32`, matching
/// `SSMWeights::l2_normalize` exactly (eps added to the sum of squares).
fn gdn_l2_normalize_scale_reference(
    x: &[f32],
    b: usize,
    seq_len: usize,
    heads: usize,
    dim: usize,
    scale: f32,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; b * seq_len * heads * dim];
    for row in 0..b * seq_len * heads {
        let xr = &x[row * dim..(row + 1) * dim];
        let sum_sq: f32 = xr.iter().map(|v| v * v).sum();
        let inv_norm = scale / (sum_sq + eps).sqrt();
        for d in 0..dim {
            out[row * dim + d] = xr[d] * inv_norm;
        }
    }
    out
}

#[test]
fn kernel_gdn_l2_normalize_scale_matches_scalar_reference() {
    let device = device();
    let kernels = Kernels::new();
    let mut rng = rng();
    fn randf(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| (rng.random::<f32>() - 0.5) * scale)
            .collect()
    }

    // (b, seq_len, heads, dim, scale) -- heads=5 (not a multiple of the
    // threadgroup width min(heads,64)) at one shape to exercise the bounds
    // check; dim=128 matches the real production state_size.
    for (b, seq_len, heads, dim, scale) in [
        (1usize, 1usize, 32usize, 128usize, 1f32 / (128f32).sqrt()),
        (2usize, 8usize, 5usize, 128usize, 1.0f32),
        (1usize, 2usize, 32usize, 128usize, 1.0f32),
    ] {
        let eps = 1e-6f32;
        let x = randf(&mut rng, b * seq_len * heads * dim, 1.0);
        let x_buf = new_buffer(&device, &x);
        let out_buf = device
            .new_buffer(
                b * seq_len * heads * dim * std::mem::size_of::<f32>(),
                RESOURCE_OPTIONS,
            )
            .unwrap();

        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        call_gdn_l2_normalize_scale_f32(
            &device,
            &encoder,
            &kernels,
            b,
            seq_len,
            heads,
            dim,
            scale,
            eps,
            &BufferOffset::zero_offset(&x_buf),
            &out_buf,
        )
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();

        let got: Vec<f32> = read_to_vec(&out_buf, b * seq_len * heads * dim);
        let expected = gdn_l2_normalize_scale_reference(&x, b, seq_len, heads, dim, scale, eps);
        let max_diff = got
            .iter()
            .zip(expected.iter())
            .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        println!("gdn_l2_normalize_scale b={b} seq_len={seq_len} heads={heads} dim={dim} scale={scale}: diff={max_diff:.8}");
        assert!(
            max_diff < 5e-5,
            "b={b} seq_len={seq_len} heads={heads}: mismatch, max diff = {max_diff}"
        );
    }
}

/// Scalar Rust reference for `kernel_gdn_decay_beta_gate_f32`, matching
/// `SSMWeights::forward`'s own `g`/`beta` computation exactly.
fn gdn_decay_beta_gate_reference(
    alpha_logits: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    beta_logits: &[f32],
    b: usize,
    seq_len: usize,
    heads: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut g = vec![0f32; b * seq_len * heads];
    let mut beta = vec![0f32; b * seq_len * heads];
    for row in 0..b * seq_len {
        for h in 0..heads {
            let idx = row * heads + h;
            // Naive log(exp(x)+1), deliberately not the more numerically
            // stable ln_1p(exp(x)) -- matches this kernel's own (and the
            // original candle code's own) naive formula exactly, so this
            // reference isn't "more correct" in a way that could show a
            // spurious diff from different rounding at the same inputs.
            let softplus = ((alpha_logits[idx] + dt_bias[h]).exp() + 1.0).ln();
            g[idx] = ssm_a[h] * softplus;
            beta[idx] = 1.0 / (1.0 + (-beta_logits[idx]).exp());
        }
    }
    (g, beta)
}

#[test]
fn kernel_gdn_decay_beta_gate_matches_scalar_reference() {
    let device = device();
    let kernels = Kernels::new();
    let mut rng = rng();
    fn randf(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| (rng.random::<f32>() - 0.5) * scale)
            .collect()
    }

    for (b, seq_len, heads) in [
        (1usize, 1usize, 32usize),
        (2usize, 8usize, 5usize),
        (1usize, 2usize, 32usize),
    ] {
        let alpha_logits = randf(&mut rng, b * seq_len * heads, 2.0);
        let dt_bias = randf(&mut rng, heads, 1.0);
        let ssm_a = randf(&mut rng, heads, 1.0);
        let beta_logits = randf(&mut rng, b * seq_len * heads, 2.0);

        let alpha_buf = new_buffer(&device, &alpha_logits);
        let dt_bias_buf = new_buffer(&device, &dt_bias);
        let ssm_a_buf = new_buffer(&device, &ssm_a);
        let beta_logits_buf = new_buffer(&device, &beta_logits);
        let g_out_buf = device
            .new_buffer(
                b * seq_len * heads * std::mem::size_of::<f32>(),
                RESOURCE_OPTIONS,
            )
            .unwrap();
        let beta_out_buf = device
            .new_buffer(
                b * seq_len * heads * std::mem::size_of::<f32>(),
                RESOURCE_OPTIONS,
            )
            .unwrap();

        let commands = commands(&device);
        let encoder = commands.command_encoder().unwrap();
        call_gdn_decay_beta_gate_f32(
            &device,
            &encoder,
            &kernels,
            b,
            seq_len,
            heads,
            &BufferOffset::zero_offset(&alpha_buf),
            &BufferOffset::zero_offset(&dt_bias_buf),
            &BufferOffset::zero_offset(&ssm_a_buf),
            &BufferOffset::zero_offset(&beta_logits_buf),
            &g_out_buf,
            &beta_out_buf,
        )
        .unwrap();
        drop(encoder);
        commands.wait_until_completed().unwrap();

        let got_g: Vec<f32> = read_to_vec(&g_out_buf, b * seq_len * heads);
        let got_beta: Vec<f32> = read_to_vec(&beta_out_buf, b * seq_len * heads);
        let (expected_g, expected_beta) = gdn_decay_beta_gate_reference(
            &alpha_logits,
            &dt_bias,
            &ssm_a,
            &beta_logits,
            b,
            seq_len,
            heads,
        );

        let g_diff = got_g
            .iter()
            .zip(expected_g.iter())
            .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        let beta_diff = got_beta
            .iter()
            .zip(expected_beta.iter())
            .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        println!("gdn_decay_beta_gate b={b} seq_len={seq_len} heads={heads}: g_diff={g_diff:.8} beta_diff={beta_diff:.8}");
        assert!(
            g_diff < 5e-5,
            "b={b} seq_len={seq_len} heads={heads}: g mismatch, max diff = {g_diff}"
        );
        assert!(
            beta_diff < 5e-5,
            "b={b} seq_len={seq_len} heads={heads}: beta mismatch, max diff = {beta_diff}"
        );
    }
}

// Regression-shaped test for the same offset bug class the earlier fused
// kernels found live: packs alpha_logits/dt_bias/ssm_a/beta_logits into
// ONE buffer with a decoy region ahead of each, at real nonzero offsets.
#[test]
fn kernel_gdn_decay_beta_gate_respects_nonzero_buffer_offsets() {
    let device = device();
    let kernels = Kernels::new();
    let commands = commands(&device);
    let encoder = commands.command_encoder().unwrap();

    let (b, seq_len, heads) = (1usize, 2usize, 8usize);
    let mut rng = rng();
    fn randf(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| (rng.random::<f32>() - 0.5) * scale)
            .collect()
    }

    let alpha_logits = randf(&mut rng, b * seq_len * heads, 2.0);
    let dt_bias = randf(&mut rng, heads, 1.0);
    let ssm_a = randf(&mut rng, heads, 1.0);
    let beta_logits = randf(&mut rng, b * seq_len * heads, 2.0);
    let decoy = randf(&mut rng, b * seq_len * heads, 999.0);

    let f32_size = std::mem::size_of::<f32>();
    let packed: Vec<f32> = [
        &decoy[..],
        &alpha_logits[..],
        &dt_bias[..],
        &ssm_a[..],
        &beta_logits[..],
    ]
    .concat();
    let packed_buf = new_buffer(&device, &packed);

    let mut offset_elems = decoy.len();
    let alpha_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };
    offset_elems += alpha_logits.len();
    let dt_bias_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };
    offset_elems += dt_bias.len();
    let ssm_a_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };
    offset_elems += ssm_a.len();
    let beta_off = BufferOffset {
        buffer: &packed_buf,
        offset_in_bytes: offset_elems * f32_size,
    };

    let g_out_buf = device
        .new_buffer(b * seq_len * heads * f32_size, RESOURCE_OPTIONS)
        .unwrap();
    let beta_out_buf = device
        .new_buffer(b * seq_len * heads * f32_size, RESOURCE_OPTIONS)
        .unwrap();

    call_gdn_decay_beta_gate_f32(
        &device,
        &encoder,
        &kernels,
        b,
        seq_len,
        heads,
        &alpha_off,
        &dt_bias_off,
        &ssm_a_off,
        &beta_off,
        &g_out_buf,
        &beta_out_buf,
    )
    .unwrap();
    drop(encoder);
    commands.wait_until_completed().unwrap();

    let got_g: Vec<f32> = read_to_vec(&g_out_buf, b * seq_len * heads);
    let got_beta: Vec<f32> = read_to_vec(&beta_out_buf, b * seq_len * heads);
    let (expected_g, expected_beta) = gdn_decay_beta_gate_reference(
        &alpha_logits,
        &dt_bias,
        &ssm_a,
        &beta_logits,
        b,
        seq_len,
        heads,
    );

    let g_diff = got_g
        .iter()
        .zip(expected_g.iter())
        .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
    let beta_diff = got_beta
        .iter()
        .zip(expected_beta.iter())
        .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
    println!("gdn_decay_beta_gate_nonzero_offsets: g_diff={g_diff:.8} beta_diff={beta_diff:.8}");
    assert!(
        g_diff < 5e-5,
        "nonzero-offset g mismatch, max diff = {g_diff}"
    );
    assert!(
        beta_diff < 5e-5,
        "nonzero-offset beta mismatch, max diff = {beta_diff}"
    );
}
