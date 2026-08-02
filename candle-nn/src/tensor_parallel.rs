//! Building blocks for Megatron-style tensor parallel linear layers.
//!
//! This module deliberately does not discover devices or create a communicator.
//! Callers provide both the devices used to shard a weight and, when needed, a
//! [`TensorReducer`] that performs their distributed reduction.

use candle::{bail, Device, Result, Tensor, D};

/// Reduces the partial outputs of a [`RowParallelLinear`] once per layer.
///
/// Implement this trait to connect a row-parallel layer to a collective such as
/// NCCL. The tensors passed to `all_reduce` live on their respective shard
/// devices and the returned tensor must live on `reduce_device`.
pub trait TensorReducer: Send + Sync {
    fn all_reduce(&self, shards: &[Tensor], reduce_device: &Device) -> Result<Tensor>;
}

/// A reducer which gathers partial outputs onto one device and sums them.
///
/// This is useful for a single device and as a portable fallback. Deployments
/// with a device-native collective should supply their own [`TensorReducer`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SumReducer;

impl TensorReducer for SumReducer {
    fn all_reduce(&self, shards: &[Tensor], reduce_device: &Device) -> Result<Tensor> {
        let Some((first, rest)) = shards.split_first() else {
            bail!("cannot reduce an empty set of tensor-parallel shards")
        };
        let mut result = first.to_device(reduce_device)?;
        for shard in rest {
            result = result.add(&shard.to_device(reduce_device)?)?;
        }
        Ok(result)
    }
}

/// A linear layer whose output features are split evenly across devices.
#[derive(Clone, Debug)]
pub struct ColumnParallelLinear {
    weights: Vec<Tensor>,
}

impl ColumnParallelLinear {
    /// Split `weight` (`[out_features, in_features]`) into equal output-feature
    /// shards, one per device.
    pub fn shard(weight: &Tensor, devices: &[Device]) -> Result<Self> {
        let (out_features, _) = weight.dims2()?;
        let shard_len = shard_len(out_features, devices.len(), "out_features")?;
        let mut weights = Vec::with_capacity(devices.len());
        for (index, device) in devices.iter().enumerate() {
            weights.push(
                weight
                    .narrow(0, index * shard_len, shard_len)?
                    .to_device(device)?
                    .contiguous()?,
            );
        }
        Ok(Self { weights })
    }

    /// Run every shard and concatenate the output features on `gather_device`.
    pub fn forward(&self, x: &Tensor, gather_device: &Device) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(self.weights.len());
        for weight in &self.weights {
            // `narrow`, `to_device`, and `t` may all produce views. CUDA matmul
            // requires contiguous operands, so make that guarantee here.
            let x = x.to_device(weight.device())?.contiguous()?;
            let weight_t = weight.t()?.contiguous()?;
            outputs.push(x.matmul(&weight_t)?.to_device(gather_device)?);
        }
        let outputs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&outputs, D::Minus1)
    }

    /// The number of devices (and weight shards) used by this layer.
    pub fn shard_count(&self) -> usize {
        self.weights.len()
    }
}

/// A linear layer whose input features are split evenly across devices.
#[derive(Clone, Debug)]
pub struct RowParallelLinear {
    weights: Vec<Tensor>,
}

impl RowParallelLinear {
    /// Split `weight` (`[out_features, in_features]`) into equal input-feature
    /// shards, one per device.
    pub fn shard(weight: &Tensor, devices: &[Device]) -> Result<Self> {
        let (_, in_features) = weight.dims2()?;
        let shard_len = shard_len(in_features, devices.len(), "in_features")?;
        let mut weights = Vec::with_capacity(devices.len());
        for (index, device) in devices.iter().enumerate() {
            weights.push(
                weight
                    .narrow(1, index * shard_len, shard_len)?
                    .to_device(device)?
                    .contiguous()?,
            );
        }
        Ok(Self { weights })
    }

    /// Run each input shard and sum the partial outputs on `reduce_device`.
    ///
    /// This uses [`SumReducer`]. Use [`Self::forward_with_reducer`] to provide
    /// a device-native all-reduce implementation.
    pub fn forward(&self, x_shards: &[Tensor], reduce_device: &Device) -> Result<Tensor> {
        self.forward_with_reducer(x_shards, reduce_device, &SumReducer)
    }

    /// Run each input shard and invoke `reducer` exactly once for the layer.
    pub fn forward_with_reducer(
        &self,
        x_shards: &[Tensor],
        reduce_device: &Device,
        reducer: &dyn TensorReducer,
    ) -> Result<Tensor> {
        if x_shards.len() != self.weights.len() {
            bail!(
                "row-parallel layer has {} weight shards but received {} input shards",
                self.weights.len(),
                x_shards.len()
            )
        }
        let mut outputs = Vec::with_capacity(self.weights.len());
        for (x, weight) in x_shards.iter().zip(&self.weights) {
            let x = x.to_device(weight.device())?.contiguous()?;
            let weight_t = weight.t()?.contiguous()?;
            outputs.push(x.matmul(&weight_t)?);
        }
        reducer.all_reduce(&outputs, reduce_device)
    }

    /// The number of devices (and weight shards) used by this layer.
    pub fn shard_count(&self) -> usize {
        self.weights.len()
    }
}

/// Split the last dimension of an activation for a [`RowParallelLinear`].
pub fn split_for_row_parallel(x: &Tensor, devices: &[Device]) -> Result<Vec<Tensor>> {
    let Some(&in_features) = x.dims().last() else {
        bail!("cannot split a scalar activation for row parallelism")
    };
    let shard_len = shard_len(in_features, devices.len(), "input features")?;
    devices
        .iter()
        .enumerate()
        .map(|(index, device)| {
            x.narrow(D::Minus1, index * shard_len, shard_len)?
                .to_device(device)?
                .contiguous()
        })
        .collect()
}

fn shard_len(features: usize, device_count: usize, name: &str) -> Result<usize> {
    if device_count == 0 {
        bail!("cannot shard {name} across an empty device list")
    }
    if !features.is_multiple_of(device_count) {
        bail!("{name} ({features}) is not divisible by the device count ({device_count})")
    }
    Ok(features / device_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weight() -> Result<Tensor> {
        Tensor::new(&[[1f32, 2., 3., 4.], [5., 6., 7., 8.]], &Device::Cpu)
    }

    #[test]
    fn column_parallel_matches_dense() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3., 4.]], &Device::Cpu)?;
        let weight = weight()?;
        let parallel = ColumnParallelLinear::shard(&weight, &[Device::Cpu, Device::Cpu])?;
        assert_eq!(
            parallel.forward(&x, &Device::Cpu)?.to_vec2::<f32>()?,
            x.matmul(&weight.t()?)?.to_vec2::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn row_parallel_matches_dense_and_reduces_once() -> Result<()> {
        struct CountingReducer(std::sync::atomic::AtomicUsize);
        impl TensorReducer for CountingReducer {
            fn all_reduce(&self, shards: &[Tensor], device: &Device) -> Result<Tensor> {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                SumReducer.all_reduce(shards, device)
            }
        }
        let x = Tensor::new(&[[1f32, 2., 3., 4.]], &Device::Cpu)?;
        let weight = weight()?;
        let devices = [Device::Cpu, Device::Cpu];
        let parallel = RowParallelLinear::shard(&weight, &devices)?;
        let reducer = CountingReducer(std::sync::atomic::AtomicUsize::new(0));
        assert_eq!(
            parallel
                .forward_with_reducer(
                    &split_for_row_parallel(&x, &devices)?,
                    &Device::Cpu,
                    &reducer
                )?
                .to_vec2::<f32>()?,
            x.matmul(&weight.t()?)?.to_vec2::<f32>()?
        );
        assert_eq!(reducer.0.load(std::sync::atomic::Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn uneven_and_empty_splits_are_errors() -> Result<()> {
        let row_uneven = Tensor::zeros((2, 3), candle::DType::F32, &Device::Cpu)?;
        let column_uneven = Tensor::zeros((3, 2), candle::DType::F32, &Device::Cpu)?;
        assert!(ColumnParallelLinear::shard(&row_uneven, &[]).is_err());
        assert!(RowParallelLinear::shard(&row_uneven, &[Device::Cpu, Device::Cpu]).is_err());
        assert!(ColumnParallelLinear::shard(&column_uneven, &[Device::Cpu, Device::Cpu]).is_err());
        Ok(())
    }
}
