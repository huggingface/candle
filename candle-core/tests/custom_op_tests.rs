use candle::cpu_backend;
use candle::{CpuStorage, CustomOp1, DType, Device, Error, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

mod test_utils;
use test_utils::to_vec1_round;

struct Elu {
    alpha: f64,
}

fn fwd<T: num_traits::Float>(v: T, alpha: T) -> T {
    if v.is_sign_positive() {
        v
    } else {
        (v.exp() - T::one()) * alpha
    }
}

impl CustomOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        use CpuStorage::*;
        let storage = match s {
            BF16(s) => {
                let alpha = bf16::from_f64(self.alpha);
                let data = cpu_backend::unary_map(s, l, |v| fwd(v, alpha));
                BF16(data)
            }
            F16(s) => {
                let alpha = f16::from_f64(self.alpha);
                let data = cpu_backend::unary_map(s, l, |v| fwd(v, alpha));
                F16(data)
            }
            F32(s) => {
                let alpha = self.alpha as f32;
                let data = cpu_backend::unary_map(s, l, |v| fwd(v, alpha));
                F32(data)
            }
            F64(s) => {
                let alpha = self.alpha;
                let data = cpu_backend::unary_map(s, l, |v| fwd(v, alpha));
                F64(data)
            }
            U8(_) => Err(Error::UnsupportedDTypeForOp(DType::U8, "elu").bt())?,
            U32(_) => Err(Error::UnsupportedDTypeForOp(DType::U32, "elu").bt())?,
        };
        Ok((storage, l.shape().clone()))
    }
}

#[test]
fn custom_op1() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = Tensor::arange(0u32, 12u32, cpu)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    let elu_t = t.custom_op1(Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round(&elu_t, 4)?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}
