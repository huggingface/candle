use candle_core::backend::BackendStorage;
use candle_core::cpu_backend;
use candle_core::test_utils::to_vec1_round;
use candle_core::{CpuStorage, CustomOp1, DType, Device, Error, Layout, Result, Shape, Tensor};

fn fwd<T>(v: T, alpha: f64) -> T
where
    T: num_traits::ToPrimitive
        + num_traits::FromPrimitive
        + num_traits::Zero
        + num_traits::One
        + Copy,
{
    let v_f64 = v.to_f64().unwrap_or(f64::NAN);
    let result = if v_f64.is_sign_positive() {
        v_f64
    } else {
        (v_f64.exp() - 1.0) * alpha
    };
    T::from_f64(result).unwrap_or_else(T::zero)
}

struct Elu {
    alpha: f64,
}

impl CustomOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let storage = candle_core::map_dtype!(
            "elu",
            s,
            |s| cpu_backend::unary_map(s, l, |v| fwd(v, self.alpha)),
            (F8E4M3, BF16, F16, F32, F64)
        );
        Ok((storage, l.shape().clone()))
    }
}

#[test]
fn custom_op1_no_backward() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = Tensor::arange(0u32, 12u32, cpu)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    let elu_t = t.apply_op1_no_bwd(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round(&elu_t, 4)?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}

// Define a similar struct as Elu but with backward support.
fn bwd<T>(v: T, alpha: f64) -> T
where
    T: num_traits::ToPrimitive
        + num_traits::FromPrimitive
        + num_traits::Zero
        + num_traits::One
        + Copy,
{
    let v_f64 = v.to_f64().unwrap_or(f64::NAN);
    let result = if v_f64.is_sign_positive() {
        1.0
    } else {
        v_f64.exp() * alpha
    };
    T::from_f64(result).unwrap_or_else(T::zero)
}

struct EluBackward {
    alpha: f64,
}

impl CustomOp1 for EluBackward {
    fn name(&self) -> &'static str {
        "elu-bwd"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let storage = candle_core::map_dtype!(
            "elu-bwd",
            s,
            |s| cpu_backend::unary_map(s, l, |v| bwd(v, self.alpha)),
            (F8E4M3, BF16, F16, F32, F64)
        );
        Ok((storage, l.shape().clone()))
    }
}

struct EluWithBackward(Elu);

impl EluWithBackward {
    fn new(alpha: f64) -> Self {
        Self(Elu { alpha })
    }
}

impl CustomOp1 for EluWithBackward {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        self.0.cpu_fwd(s, l)
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let alpha = self.0.alpha;
        let bwd = arg.apply_op1(EluBackward { alpha })?;
        Ok(Some(grad_res.mul(&bwd)?))
    }
}

#[test]
fn custom_op1_with_backward() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = candle_core::Var::new(&[-2f32, 0f32, 2f32], cpu)?;
    let elu_t = t.apply_op1(EluWithBackward::new(2.))?;
    assert_eq!(to_vec1_round(&elu_t, 4)?, &[-1.7293, 0.0, 2.0]);

    let grads = elu_t.backward()?;
    let grad_x = grads.get(&t).unwrap();
    assert_eq!(to_vec1_round(grad_x, 4)?, [0.2707, 1.0, 1.0]);

    Ok(())
}

impl candle_core::InplaceOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        let alpha = self.alpha;
        match s {
            CpuStorage::F8E4M3(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            CpuStorage::BF16(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            CpuStorage::F16(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            CpuStorage::F32(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            CpuStorage::F64(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            _ => candle_core::bail!("unsupported dtype for inplace elu"),
        }
        Ok(())
    }
}

#[test]
fn inplace_op1() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = Tensor::arange(0u32, 12u32, cpu)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    t.inplace_op1(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round(&t, 4)?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}

struct InplaceAdd;

impl candle_core::InplaceOp2 for InplaceAdd {
    fn name(&self) -> &'static str {
        "inplace-add"
    }

    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<()> {
        let elems = l1.shape().elem_count();
        let start1 = l1.start_offset();
        let start2 = l2.start_offset();
        let (CpuStorage::F32(d1), CpuStorage::F32(d2)) = (s1, s2) else {
            candle_core::bail!("unsupported dtype for inplace-add")
        };
        let rhs: Vec<f32> = d2[start2..start2 + elems].to_vec();
        for (v, r) in d1[start1..start1 + elems].iter_mut().zip(rhs) {
            *v += r;
        }
        Ok(())
    }
}

impl candle_core::InplaceOp3 for InplaceAdd {
    fn name(&self) -> &'static str {
        "inplace-add3"
    }

    fn cpu_fwd(
        &self,
        _: &mut CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<()> {
        Ok(())
    }
}

#[test]
fn inplace_op_shared_storage_errors_instead_of_deadlocking() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = Tensor::from_vec(vec![1f32, 2., 3., 4.], 4, cpu)?;
    let dst = t.narrow(0, 0, 2)?;
    let src = t.narrow(0, 2, 2)?;
    let res = dst.inplace_op2(&src, &InplaceAdd);
    assert!(res.is_err(), "shared-storage inplace_op2 must error");
    let res = dst.inplace_op3(&src, &src, &InplaceAdd);
    let _ = res.expect_err("shared-storage inplace_op3 must error");
    // Distinct storages keep working.
    let a = Tensor::from_vec(vec![1f32, 2.], 2, cpu)?;
    let b = Tensor::from_vec(vec![10f32, 20.], 2, cpu)?;
    a.inplace_op2(&b, &InplaceAdd)?;
    assert_eq!(a.to_vec1::<f32>()?, [11., 22.]);
    Ok(())
}
