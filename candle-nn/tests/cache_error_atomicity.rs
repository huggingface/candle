//! Private qualification, CPU only. No production changes.
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{ConcatKvCache, KvCache, RotatingCache, RotatingKvCache};

fn tensor(data: &[f32], width: usize) -> Result<Tensor> {
    Tensor::from_slice(data, (data.len() / width, width), &Device::Cpu)
}
fn values(t: &Tensor) -> Result<Vec<Vec<f32>>> {
    t.to_vec2::<f32>()
}
fn seed() -> Result<(Tensor, Tensor)> {
    Ok((tensor(&[1., 2.], 2)?, tensor(&[10., 20., 30.], 3)?))
}
fn next() -> Result<(Tensor, Tensor)> {
    Ok((tensor(&[3., 4.], 2)?, tensor(&[40., 50., 60.], 3)?))
}

#[test]
fn preallocated_error_preserves_lengths_and_values() -> Result<()> {
    let (k, v) = seed()?;
    let (nk, nv) = next()?;
    for bad in [tensor(&[7., 8., 9., 10.], 4)?, nv.to_dtype(DType::U32)?] {
        let mut c = KvCache::new(0, 4);
        c.append(&k, &v)?;
        let before = (
            c.k_cache().current_seq_len(),
            c.v_cache().current_seq_len(),
            values(&c.k()?.unwrap())?,
            values(&c.v()?.unwrap())?,
        );
        assert!(c.append(&nk, &bad).is_err());
        let after = (
            c.k_cache().current_seq_len(),
            c.v_cache().current_seq_len(),
            values(&c.k()?.unwrap())?,
            values(&c.v()?.unwrap())?,
        );
        assert_eq!(
            after, before,
            "failed paired append changed visible cache state"
        );
    }
    Ok(())
}
#[test]
fn preallocated_valid_retry_matches_fresh_cache() -> Result<()> {
    let (k, v) = seed()?;
    let (nk, nv) = next()?;
    let mut c = KvCache::new(0, 4);
    c.append(&k, &v)?;
    assert!(c.append(&nk, &tensor(&[7., 8., 9., 10.], 4)?).is_err());
    let actual = c.append(&nk, &nv)?;
    let mut fresh = KvCache::new(0, 4);
    fresh.append(&k, &v)?;
    let expected = fresh.append(&nk, &nv)?;
    assert_eq!(
        (values(&actual.0)?, values(&actual.1)?),
        (values(&expected.0)?, values(&expected.1)?)
    );
    Ok(())
}
#[test]
fn concat_error_preserves_values() -> Result<()> {
    let (k, v) = seed()?;
    let (nk, nv) = next()?;
    for bad in [tensor(&[7., 8., 9., 10.], 4)?, nv.to_dtype(DType::U32)?] {
        let mut c = ConcatKvCache::new(0);
        c.append(&k, &v)?;
        let before = (values(c.k().unwrap())?, values(c.v().unwrap())?);
        assert!(c.append(&nk, &bad).is_err());
        assert_eq!((values(c.k().unwrap())?, values(c.v().unwrap())?), before);
    }
    Ok(())
}
#[test]
fn concat_valid_retry_matches_fresh_cache() -> Result<()> {
    let (k, v) = seed()?;
    let (nk, nv) = next()?;
    let mut c = ConcatKvCache::new(0);
    c.append(&k, &v)?;
    assert!(c.append(&nk, &tensor(&[7., 8., 9., 10.], 4)?).is_err());
    let actual = c.append(&nk, &nv)?;
    let mut fresh = ConcatKvCache::new(0);
    fresh.append(&k, &v)?;
    let expected = fresh.append(&nk, &nv)?;
    assert_eq!(
        (values(&actual.0)?, values(&actual.1)?),
        (values(&expected.0)?, values(&expected.1)?)
    );
    Ok(())
}
#[test]
fn rotating_pair_error_preserves_metadata_and_values() -> Result<()> {
    let (k, v) = seed()?;
    let (nk, _) = next()?;
    let mut c = RotatingKvCache::new(0, 4);
    c.append(&k, &v)?;
    let before = (
        c.k_cache().current_seq_len(),
        c.v_cache().current_seq_len(),
        c.k_cache().offset(),
        c.v_cache().offset(),
        values(&c.k()?.unwrap())?,
        values(&c.v()?.unwrap())?,
    );
    assert!(c.append(&nk, &tensor(&[7., 8., 9., 10.], 4)?).is_err());
    let after = (
        c.k_cache().current_seq_len(),
        c.v_cache().current_seq_len(),
        c.k_cache().offset(),
        c.v_cache().offset(),
        values(&c.k()?.unwrap())?,
        values(&c.v()?.unwrap())?,
    );
    assert_eq!(after, before);
    Ok(())
}
#[test]
fn rotating_single_error_preserves_metadata_and_values() -> Result<()> {
    let (k, _) = seed()?;
    let mut c = RotatingCache::new(0, 4);
    c.append(&k)?;
    let before = (
        c.current_seq_len(),
        c.offset(),
        values(&c.current_data()?.unwrap())?,
    );
    assert!(c.append(&tensor(&[7., 8., 9.], 3)?).is_err());
    assert_eq!(
        (
            c.current_seq_len(),
            c.offset(),
            values(&c.current_data()?.unwrap())?
        ),
        before
    );
    Ok(())
}
#[test]
fn successful_different_key_value_widths_are_supported() -> Result<()> {
    let (k, v) = seed()?;
    let (nk, nv) = next()?;
    let mut a = KvCache::new(0, 4);
    a.append(&k, &v)?;
    let av = a.append(&nk, &nv)?;
    let mut b = ConcatKvCache::new(0);
    b.append(&k, &v)?;
    let bv = b.append(&nk, &nv)?;
    let mut c = RotatingKvCache::new(0, 4);
    c.append(&k, &v)?;
    let cv = c.append(&nk, &nv)?;
    for (k, v) in [av, bv, cv] {
        assert_eq!(k.dims(), &[2, 2]);
        assert_eq!(v.dims(), &[2, 3]);
        assert_eq!(values(&k)?, vec![vec![1., 2.], vec![3., 4.]]);
        assert_eq!(values(&v)?, vec![vec![10., 20., 30.], vec![40., 50., 60.]]);
    }
    Ok(())
}

#[test]
fn preallocated_rejects_noncontiguous_value_before_key_change() -> Result<()> {
    let (k, v) = seed()?;
    let (nk, _) = next()?;
    let bad =
        Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], (3, 2), &Device::Cpu)?.transpose(0, 1)?;
    assert!(!bad.is_contiguous());
    let mut c = KvCache::new(0, 4);
    c.append(&k, &v)?;
    assert!(c.append(&nk, &bad).is_err());
    assert_eq!(c.k_cache().current_seq_len(), 1);
    assert_eq!(c.v_cache().current_seq_len(), 1);
    assert_eq!(values(&c.k()?.unwrap())?, values(&k)?);
    Ok(())
}
#[test]
fn preallocated_rejects_bad_value_rank_before_key_change() -> Result<()> {
    let (k, v) = seed()?;
    let (nk, _) = next()?;
    let mut c = KvCache::new(0, 4);
    c.append(&k, &v)?;
    let bad = Tensor::from_slice(&[1f32, 2., 3.], (1, 1, 3), &Device::Cpu)?;
    assert!(c.append(&nk, &bad).is_err());
    assert_eq!(c.k_cache().current_seq_len(), 1);
    assert_eq!(c.v_cache().current_seq_len(), 1);
    Ok(())
}
#[test]
fn rotating_wrap_rejects_bad_value_before_overwriting_keys() -> Result<()> {
    let (k, v) = seed()?;
    let (nk, nv) = next()?;
    let mut c = RotatingKvCache::new(0, 2);
    c.append(&k, &v)?;
    c.append(&nk, &nv)?;
    let before = (
        values(&c.k()?.unwrap())?,
        values(&c.v()?.unwrap())?,
        c.current_seq_len(),
        c.offset(),
    );
    assert!(c.append(&nk, &tensor(&[1., 2., 3., 4.], 4)?).is_err());
    assert_eq!(
        (
            values(&c.k()?.unwrap())?,
            values(&c.v()?.unwrap())?,
            c.current_seq_len(),
            c.offset()
        ),
        before
    );
    Ok(())
}
#[test]
fn failed_first_key_does_not_initialize_preallocated_cache() -> Result<()> {
    let mut c = KvCache::new(2, 4);
    let (k, v) = seed()?;
    assert!(c.append(&k, &v).is_err());
    assert!(c.k_cache().all_data().is_none());
    assert!(c.v_cache().all_data().is_none());
    Ok(())
}
#[test]
fn dtype_error_is_checked_independently() -> Result<()> {
    let (k, v) = seed()?;
    let (nk, nv) = next()?;
    let mut a = KvCache::new(0, 4);
    a.append(&k, &v)?;
    assert!(a.append(&nk, &nv.to_dtype(DType::U32)?).is_err());
    assert_eq!(a.k_cache().current_seq_len(), 1);
    let mut b = ConcatKvCache::new(0);
    b.append(&k, &v)?;
    assert!(b.append(&nk, &nv.to_dtype(DType::U32)?).is_err());
    assert_eq!(b.current_seq_len(), 1);
    Ok(())
}
