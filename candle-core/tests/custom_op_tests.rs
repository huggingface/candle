use candle_core::backend::BackendStorage;
use candle_core::cpu_backend::{self, CpuDevice};
use candle_core::test_utils::to_vec1_round;
use candle_core::{CpuStorage, CustomOp1, DType, Error, Layout, Result, Shape, Tensor};

type CpuTensor = Tensor<CpuStorage>;

fn fwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        v
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        (v.exp() - T::one()) * alpha
    }
}

struct Elu {
    alpha: f64,
}

impl CustomOp1<CpuStorage> for Elu {
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
    let t = CpuTensor::arange(0u32, 12u32, &CpuDevice)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    let elu_t = t.apply_op1_no_bwd(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round(&elu_t, 4)?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}

// Define a similar struct as Elu but with backward support.
fn bwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        T::one()
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        v.exp() * alpha
    }
}

struct EluBackward {
    alpha: f64,
}

impl CustomOp1<CpuStorage> for EluBackward {
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

impl CustomOp1<CpuStorage> for EluWithBackward {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        self.0.cpu_fwd(s, l)
    }

    fn bwd(
        &self,
        arg: &CpuTensor,
        _res: &CpuTensor,
        grad_res: &CpuTensor,
    ) -> Result<Option<CpuTensor>> {
        let alpha = self.0.alpha;
        let bwd = arg.apply_op1(EluBackward { alpha })?;
        Ok(Some(grad_res.mul(&bwd)?))
    }
}

#[test]
fn custom_op1_with_backward() -> Result<()> {
    let t = candle_core::Var::new(&[-2f32, 0f32, 2f32], &CpuDevice)?;
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
    let t = CpuTensor::arange(0u32, 12u32, &CpuDevice)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    t.inplace_op1(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round(&t, 4)?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[allow(clippy::approx_constant)]
#[test]
fn ug_op() -> Result<()> {
    use candle_core::backend::{BackendDevice, UgDevice};
    use candle_core::{InplaceOp1, UgIOp1};

    fn inner<B: BackendStorage>(device: &B::Device) -> Result<()>
    where
        B::Device: UgDevice,
        UgIOp1<B>: InplaceOp1,
    {
        let kernel = {
            use ug::lang::op;

            let layout = ug::Layout::from_shape(&[12]);
            let ptr = op::Arg::ptr(ug::DType::F32);
            let src = op::load(ptr.id(), layout.clone(), ug::DType::F32)?;
            let src = op::unary(op::UnaryOp::Exp, src)?;
            let st = op::store(ptr.id(), layout, src)?;
            let kernel = op::Kernel::new("exp".to_string(), vec![ptr], vec![st]);
            let opts: ug::lower_op::Opts = Default::default();
            kernel.lower(&opts)?
        };
        let op = candle_core::UgIOp1::<B>::new("test", kernel, device)?;
        let t: Tensor<B> = Tensor::arange(0u32, 12u32, device)?.to_dtype(DType::F32)?;
        t.inplace_op1(&op)?;
        assert_eq!(
            to_vec1_round(&t, 2)?,
            &[
                1.0, 2.72, 7.39, 20.09, 54.6, 148.41, 403.43, 1096.63, 2980.96, 8103.08, 22026.47,
                59874.13
            ]
        );
        Ok(())
    }
    #[cfg(feature = "cuda")]
    {
        use candle_core::{CudaDevice, CudaStorage};
        inner::<CudaStorage>(&CudaDevice::new(0)?)?;
    }
    #[cfg(feature = "metal")]
    {
        use candle_core::{MetalDevice, MetalStorage};
        inner::<MetalStorage>(&MetalDevice::new(0)?)?;
    }

    Ok(())
}
