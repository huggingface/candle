use candle::{Result, Tensor};

pub fn pad_and_stack_1d(items: &[Tensor], pad_value: f64) -> Result<(Tensor, Tensor)> {
    pad_and_stack_impl(items, pad_value, 1, "pad_and_stack_1d")
}

pub fn pad_and_stack_2d(items: &[Tensor], pad_value: f64) -> Result<(Tensor, Tensor)> {
    pad_and_stack_impl(items, pad_value, 2, "pad_and_stack_2d")
}

fn pad_and_stack_impl(
    items: &[Tensor],
    pad_value: f64,
    expected_rank: usize,
    fn_name: &str,
) -> Result<(Tensor, Tensor)> {
    if items.is_empty() {
        candle::bail!("{fn_name}: empty batch");
    }
    for (i, t) in items.iter().enumerate() {
        if t.dims().len() != expected_rank {
            candle::bail!(
                "{fn_name}: item {i} has rank {}, expected {expected_rank}",
                t.dims().len()
            );
        }
    }
    let device = items[0].device().clone();
    let dtype = items[0].dtype();
    let feature_dim = if expected_rank == 2 {
        items[0].dims()[1]
    } else {
        0
    };

    for (i, t) in items.iter().enumerate() {
        if expected_rank == 2 && t.dims()[1] != feature_dim {
            candle::bail!(
                "{fn_name}: item {i} has feature dim {}, expected {feature_dim}",
                t.dims()[1]
            );
        }
        if t.dtype() != dtype {
            candle::bail!(
                "{fn_name}: item {i} has dtype {:?}, expected {:?}",
                t.dtype(),
                dtype
            );
        }
        if !t.device().same_device(&device) {
            candle::bail!("{fn_name}: item {i} is on a different device");
        }
    }

    let lengths: Vec<usize> = items.iter().map(|t| t.dims()[0]).collect();
    let t_max = *lengths.iter().max().unwrap();
    let b = items.len();

    let mut padded_rows: Vec<Tensor> = Vec::with_capacity(b);
    for (t, &len) in items.iter().zip(lengths.iter()) {
        let row = if len == t_max {
            t.clone()
        } else {
            let pad_len = t_max - len;
            let pad = if expected_rank == 1 {
                Tensor::full(pad_value, pad_len, &device)?.to_dtype(dtype)?
            } else {
                Tensor::full(pad_value, (pad_len, feature_dim), &device)?.to_dtype(dtype)?
            };
            Tensor::cat(&[t, &pad], 0)?
        };
        padded_rows.push(row);
    }
    let padded = Tensor::stack(&padded_rows, 0)?;

    let mut mask_data = vec![0u8; b * t_max];
    for (i, &len) in lengths.iter().enumerate() {
        for v in mask_data[i * t_max..i * t_max + len].iter_mut() {
            *v = 1;
        }
    }
    let mask = Tensor::from_vec(mask_data, (b, t_max), &device)?;

    Ok((padded, mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn pad_1d_basic_shape() {
        let dev = Device::Cpu;
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap();
        let b = Tensor::new(&[4.0f32, 5.0], &dev).unwrap();
        let c = Tensor::new(&[6.0f32], &dev).unwrap();
        let (p, m) = pad_and_stack_1d(&[a, b, c], 0.0).unwrap();
        assert_eq!(p.dims(), &[3, 3]);
        assert_eq!(m.dims(), &[3, 3]);
        assert_eq!(p.dtype(), DType::F32);
        assert_eq!(m.dtype(), DType::U8);

        let p_vec: Vec<Vec<f32>> = p.to_vec2().unwrap();
        assert_eq!(p_vec[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(p_vec[1], vec![4.0, 5.0, 0.0]);
        assert_eq!(p_vec[2], vec![6.0, 0.0, 0.0]);

        let m_vec: Vec<Vec<u8>> = m.to_vec2().unwrap();
        assert_eq!(m_vec[0], vec![1, 1, 1]);
        assert_eq!(m_vec[1], vec![1, 1, 0]);
        assert_eq!(m_vec[2], vec![1, 0, 0]);
    }

    #[test]
    fn pad_1d_equal_lengths_no_padding() {
        let dev = Device::Cpu;
        let a = Tensor::new(&[1.0f32, 2.0], &dev).unwrap();
        let b = Tensor::new(&[3.0f32, 4.0], &dev).unwrap();
        let (p, m) = pad_and_stack_1d(&[a, b], -1.0).unwrap();
        assert_eq!(p.dims(), &[2, 2]);
        let m_vec: Vec<Vec<u8>> = m.to_vec2().unwrap();
        assert_eq!(m_vec, vec![vec![1, 1], vec![1, 1]]);
    }

    #[test]
    fn pad_1d_custom_pad_value() {
        let dev = Device::Cpu;
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap();
        let b = Tensor::new(&[4.0f32], &dev).unwrap();
        let (p, _) = pad_and_stack_1d(&[a, b], -99.0).unwrap();
        let p_vec: Vec<Vec<f32>> = p.to_vec2().unwrap();
        assert_eq!(p_vec[1], vec![4.0, -99.0, -99.0]);
    }

    #[test]
    fn pad_1d_empty_batch_errors() {
        let err = pad_and_stack_1d(&[], 0.0).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("empty batch"), "unexpected error: {msg}");
    }

    #[test]
    fn pad_1d_wrong_rank_errors() {
        let dev = Device::Cpu;
        let bad = Tensor::zeros((2, 3), DType::F32, &dev).unwrap();
        let err = pad_and_stack_1d(&[bad], 0.0).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("rank"), "unexpected error: {msg}");
    }

    #[test]
    fn pad_2d_basic_shape() {
        let dev = Device::Cpu;
        let a = Tensor::arange(0.0f32, 6.0, &dev)
            .unwrap()
            .reshape((3, 2))
            .unwrap();
        let b = Tensor::arange(10.0f32, 14.0, &dev)
            .unwrap()
            .reshape((2, 2))
            .unwrap();
        let (p, m) = pad_and_stack_2d(&[a, b], 0.0).unwrap();
        assert_eq!(p.dims(), &[2, 3, 2]);
        assert_eq!(m.dims(), &[2, 3]);

        let m_vec: Vec<Vec<u8>> = m.to_vec2().unwrap();
        assert_eq!(m_vec[0], vec![1, 1, 1]);
        assert_eq!(m_vec[1], vec![1, 1, 0]);

        let padded_b: Vec<Vec<f32>> = p.get(1).unwrap().to_vec2().unwrap();
        assert_eq!(padded_b[2], vec![0.0, 0.0]);
    }

    #[test]
    fn pad_2d_mismatched_feature_dim_errors() {
        let dev = Device::Cpu;
        let a = Tensor::zeros((3, 4), DType::F32, &dev).unwrap();
        let b = Tensor::zeros((3, 5), DType::F32, &dev).unwrap();
        let err = pad_and_stack_2d(&[a, b], 0.0).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("feature dim"), "unexpected error: {msg}");
    }

    #[test]
    fn pad_2d_mismatched_dtype_errors() {
        let dev = Device::Cpu;
        let a = Tensor::zeros((3, 4), DType::F32, &dev).unwrap();
        let b = Tensor::zeros((2, 4), DType::F64, &dev).unwrap();
        let err = pad_and_stack_2d(&[a, b], 0.0).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("dtype"), "unexpected error: {msg}");
    }
}
