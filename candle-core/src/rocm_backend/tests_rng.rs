//! Tests for the rocrand-backed `rand_uniform` / `rand_normal` launchers.

use super::tests::rocm_device;
use crate::backend::BackendStorage;
use crate::{CpuStorage, DType, Device, Result, Shape, Tensor};
use half::{bf16, f16};

fn cpu_storage_len(storage: &CpuStorage) -> usize {
    match storage {
        CpuStorage::F32(v) => v.len(),
        CpuStorage::F64(v) => v.len(),
        other => panic!("unexpected cpu storage {:?}", other.dtype()),
    }
}

/// rocrand's normal generators need an even element count, so an odd shape is
/// generated one element long. Returning that buffer as-is made
/// `to_cpu_storage` hand back one element more than the shape has, because
/// `clone_dtoh` sizes its host `Vec` from the device allocation.
#[test]
fn rand_normal_with_an_odd_element_count_matches_the_shape() -> Result<()> {
    let dev = rocm_device!();
    let Device::Rocm(rocm) = &dev else {
        panic!("expected a ROCm device")
    };

    for elem_count in [1usize, 7, 15] {
        let shape = Shape::from(elem_count);
        for dtype in [DType::F32, DType::F64] {
            let storage = rocm.rand_normal_impl(&shape, dtype, 0., 1.)?;
            assert_eq!(storage.dtype(), dtype);
            assert_eq!(
                cpu_storage_len(&storage.to_cpu_storage()?),
                elem_count,
                "{dtype:?} storage length for {elem_count} elements"
            );
        }
    }

    // Even counts must keep working untouched.
    let storage = rocm.rand_normal_impl(&Shape::from(8usize), DType::F32, 0., 1.)?;
    assert_eq!(cpu_storage_len(&storage.to_cpu_storage()?), 8);
    Ok(())
}

#[test]
fn rand_uniform_f16_and_bf16() -> Result<()> {
    let dev = rocm_device!();

    let t = Tensor::rand(f16::from_f32(-2.), f16::from_f32(3.), (3, 5), &dev)?;
    assert_eq!(t.dtype(), DType::F16);
    let values = t.flatten_all()?.to_vec1::<f16>()?;
    assert_eq!(values.len(), 15);
    for v in values {
        let v = v.to_f32();
        assert!((-2.0..=3.0).contains(&v), "{v} out of range");
    }

    let t = Tensor::rand(bf16::from_f32(1.), bf16::from_f32(4.), (2, 6), &dev)?;
    assert_eq!(t.dtype(), DType::BF16);
    let values = t.flatten_all()?.to_vec1::<bf16>()?;
    assert_eq!(values.len(), 12);
    for v in values {
        let v = v.to_f32();
        // bf16 has 8 mantissa bits, so the cast can round a value just past the
        // bound; allow a single ulp of slack at this magnitude.
        assert!((0.98..=4.02).contains(&v), "{v} out of range");
    }
    Ok(())
}

#[test]
fn rand_normal_f16_and_bf16() -> Result<()> {
    let dev = rocm_device!();

    // 21 elements: odd, so it also covers the even-count rounding in half
    // precision.
    let t = Tensor::randn(f16::from_f32(0.), f16::from_f32(1.), (3, 7), &dev)?;
    assert_eq!(t.dtype(), DType::F16);
    let values = t.flatten_all()?.to_vec1::<f16>()?;
    assert_eq!(values.len(), 21);
    for v in values {
        let v = v.to_f32();
        assert!(
            v.is_finite() && v.abs() < 10.,
            "{v} is not a plausible draw"
        );
    }

    let t = Tensor::randn(bf16::from_f32(5.), bf16::from_f32(0.5), (11,), &dev)?;
    assert_eq!(t.dtype(), DType::BF16);
    let values = t.to_vec1::<bf16>()?;
    assert_eq!(values.len(), 11);
    for v in values {
        let v = v.to_f32();
        assert!(
            v.is_finite() && (0.0..10.).contains(&v),
            "{v} is far off mean"
        );
    }
    Ok(())
}
