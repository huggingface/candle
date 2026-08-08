//! Loading NVIDIA ModelOpt NVFP4 (W4A16) checkpoints from safetensors.
//!
//! NVFP4 checkpoints store each packed linear weight as three tensors
//! instead of one dense weight:
//!   - the packed weight: a `U8` tensor of shape `[out_dim, in_dim / 2]`,
//!     packing two E2M1 values per byte (low then high nibble). ModelOpt
//!     exports this as `.weight` (e.g. `nvidia/Llama-3.3-70B-Instruct-FP4`);
//!     compressed-tensors checkpoints (llm-compressor / vLLM) use
//!     `.weight_packed` instead. See [`nvfp4_weight_tensor_name`].
//!   - `weight_scale`: an `F8E4M3` tensor of shape `[out_dim, in_dim / block_size]`,
//!     one scale per `block_size`-wide block of the input dimension
//!     (`block_size` is 16 for NVFP4).
//!   - `weight_scale_2`: an `f32` tensor holding exactly one element (rank 0
//!     or `[1]` depending on the exporter) that scales the whole matrix.
//!     Its presence is the unambiguous signal that a layer is NVFP4-backed;
//!     see [`detect_backing`].
//!
//! `weight[i, j] = e2m1(packed[i, j]) * e4m3(weight_scale[i, j / block_size]) * weight_scale_2`.
//!
//! Activations are not quantized in W4A16, so an `input_scale` tensor some
//! checkpoints also carry alongside these three is intentionally never read
//! here.
//!
//! This module never reinterprets the packed bytes as `f32`: [`dequantize_nvfp4`]
//! decodes each nibble and scale byte explicitly through [`unpack_e2m1`] and
//! `float8::F8E4M3::to_f32`, and [`nvfp4_linear`] refuses to materialize a
//! dense tensor larger than an explicit byte budget. See the `nvfp4-cuda`
//! feature's [`cuda`] submodule for a path that keeps the packed weight
//! device-resident instead of expanding it 8x into a dense `f32` tensor.
//!
//! [`auto_linear`] picks among three execution paths, in this order
//! (huggingface/candle#3857 -- NVFP4 had no execution path off CUDA):
//!
//!   1. `nvfp4-cuda` fused kernel ([`cuda::Nvfp4LinearCuda`]): packed and
//!      device-resident, used only on a CUDA device with the kernel
//!      available.
//!   2. [`Nvfp4LinearPacked`]: a CPU operator that keeps the weight in its
//!      packed form (~1x the checkpoint's footprint, not 8x) and decodes
//!      each output row on the fly during `forward`, trading some compute
//!      for memory. This is the default off CUDA -- on a CPU host or Apple
//!      Silicon, where the dense fallback below can OOM or refuse to run,
//!      memory is the actual blocker, not speed. There is no Metal kernel
//!      yet; on a Metal device this path still round-trips through host
//!      memory rather than running on-device.
//!   3. [`nvfp4_linear`]: dequantizes to a dense tensor once at load time.
//!      Opt in via `Nvfp4Config::force_dense_fallback` when raw forward-pass
//!      throughput matters more than resident memory and the checkpoint
//!      fits within `max_dense_bytes`.

use candle::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// Number of input columns sharing one `weight_scale` byte.
pub const BLOCK_SIZE: usize = 16;

/// What backs a linear layer's weight in a checkpoint that may mix dense,
/// block-wise FP8, and packed NVFP4 layers (as ModelOpt exports do).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backing {
    /// A plain dense `weight` tensor.
    Dense,
    /// A block-wise FP8 weight (`weight` + `weight_scale_inv`), e.g. as used
    /// by DeepSeek-V3-style checkpoints.
    Fp8Block,
    /// A packed NVFP4 weight: `weight_scale` + `weight_scale_2`, plus the
    /// packed weight itself under `.weight` (ModelOpt) or `.weight_packed`
    /// (compressed-tensors) -- see [`nvfp4_weight_tensor_name`].
    Nvfp4,
}

/// Dequantizes a block-wise FP8 weight into a dense `[out_dim, in_dim]` `f32`
/// tensor. `weight` is an `F8E4M3` tensor of shape `[out_dim, in_dim]` and
/// `scale` is an `f32` tensor of shape
/// `[ceil(out_dim / block_size), ceil(in_dim / block_size)]` holding one
/// scale per `block_size x block_size` block of `weight`.
pub fn dequantize_fp8_block(weight: &Tensor, scale: &Tensor, block_size: usize) -> Result<Tensor> {
    let (out_dim, in_dim) = weight.dims2()?;
    let (scale_rows, scale_cols) = scale.dims2()?;
    if scale_rows != out_dim.div_ceil(block_size) || scale_cols != in_dim.div_ceil(block_size) {
        candle::bail!(
            "fp8: scale shape {:?} does not match weight shape {:?} for block_size {block_size}",
            (scale_rows, scale_cols),
            (out_dim, in_dim),
        )
    }
    let weight = weight.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let scale = scale.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut out = vec![0f32; out_dim * in_dim];
    for (row, weight_row) in weight.iter().enumerate() {
        let scale_row = &scale[row / block_size];
        for (col, &w) in weight_row.iter().enumerate() {
            out[row * in_dim + col] = w * scale_row[col / block_size];
        }
    }
    Tensor::from_vec(out, (out_dim, in_dim), &candle::Device::Cpu)
}

fn fp8_block_linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    block_size: usize,
) -> Result<Linear> {
    let weight = vb.get_with_hints_dtype(
        (out_dim, in_dim),
        "weight",
        Default::default(),
        DType::F8E4M3,
    )?;
    let scale = vb.get_with_hints_dtype(
        (out_dim.div_ceil(block_size), in_dim.div_ceil(block_size)),
        "weight_scale_inv",
        Default::default(),
        DType::F32,
    )?;
    let weight = dequantize_fp8_block(&weight, &scale, block_size)?
        .to_dtype(vb.dtype())?
        .to_device(vb.device())?;
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

/// Inspects the tensor names available at `vb`'s current path to determine
/// what backs the linear layer, without reading any tensor data.
///
/// `weight_scale_2` alone is the NVFP4 signal: both the ModelOpt and
/// compressed-tensors export conventions include it, and neither a dense nor
/// a block-wise-FP8 layer has one. The packed weight's own tensor name
/// differs between the two conventions -- see [`nvfp4_weight_tensor_name`].
pub fn detect_backing(vb: &VarBuilder) -> Backing {
    if vb.contains_tensor("weight_scale_2") {
        Backing::Nvfp4
    } else if vb.contains_tensor("weight_scale_inv") {
        Backing::Fp8Block
    } else {
        Backing::Dense
    }
}

/// The tensor name holding the packed NVFP4 weight at `vb`'s current path:
/// `weight_packed` under the compressed-tensors convention (llm-compressor /
/// vLLM) if present, otherwise `weight` -- the ModelOpt convention, e.g.
/// `nvidia/Llama-3.3-70B-Instruct-FP4`, which does not use `weight_packed`
/// at all. Only meaningful once [`detect_backing`] has returned
/// [`Backing::Nvfp4`].
pub fn nvfp4_weight_tensor_name(vb: &VarBuilder) -> &'static str {
    if vb.contains_tensor("weight_packed") {
        "weight_packed"
    } else {
        "weight"
    }
}

/// Reads `name` as an `f32` tensor holding exactly one element, accepting
/// either a rank-0 scalar or a rank-1 `[1]` tensor: ModelOpt exporters have
/// used both shapes for `weight_scale_2` across versions.
///
/// Tries the shapes with ordinary, shape-checked reads rather than
/// `get_unchecked`, since some `SimpleBackend`s reject it outright (e.g.
/// `VarMap`, used to build a randomly-initialized `VarBuilder`). Note this
/// module still isn't reachable from a `ShardedVarBuilder` (used for
/// tensor-parallel loading): that's a different `VarBuilderArgs<B>`
/// instantiation than the concrete `VarBuilder` these functions take, so
/// it's a type-level gap this fix doesn't close.
fn read_scalar_f32(vb: &VarBuilder, name: &str) -> Result<f32> {
    if let Ok(t) = vb.get_with_hints_dtype((), name, Default::default(), DType::F32) {
        return t.to_scalar::<f32>();
    }
    let t = vb.get_with_hints_dtype(1, name, Default::default(), DType::F32)?;
    t.reshape(())?.to_scalar::<f32>()
}

fn unpack_e2m1(nibble: u8) -> f32 {
    const TABLE: [f32; 16] = [
        0., 0.5, 1., 1.5, 2., 3., 4., 6., 0., -0.5, -1., -1.5, -2., -3., -4., -6.,
    ];
    TABLE[(nibble & 0xf) as usize]
}

/// Dequantizes a packed NVFP4 weight into a dense `[out_dim, in_dim]` `f32`
/// tensor on the CPU.
///
/// `packed_weight` is a `U8` tensor of shape `[out_dim, in_dim / 2]` and
/// `weight_scale` is an `F8E4M3` tensor of shape
/// `[out_dim, in_dim / block_size]`. Fails without allocating if the
/// resulting dense tensor would exceed `max_dense_bytes`, since unpacking
/// NVFP4 expands its footprint roughly 8x (4-bit weights plus one scale byte
/// per `block_size` values, versus 32-bit floats).
pub fn dequantize_nvfp4(
    packed_weight: &Tensor,
    weight_scale: &Tensor,
    weight_scale_2: f32,
    block_size: usize,
    max_dense_bytes: usize,
) -> Result<Tensor> {
    let (out_dim, packed_in_dim) = packed_weight.dims2()?;
    let in_dim = packed_in_dim * 2;
    let (scale_rows, scale_cols) = weight_scale.dims2()?;
    if scale_rows != out_dim || scale_cols != in_dim.div_ceil(block_size) {
        candle::bail!(
            "nvfp4: weight_scale shape {:?} does not match packed weight shape {:?} for block_size {block_size}",
            (scale_rows, scale_cols),
            (out_dim, in_dim),
        )
    }
    if packed_weight.dtype() != DType::U8 {
        candle::bail!(
            "nvfp4: weight_packed must be u8, got {:?}",
            packed_weight.dtype()
        )
    }
    if weight_scale.dtype() != DType::F8E4M3 {
        candle::bail!(
            "nvfp4: weight_scale must be f8e4m3, got {:?}",
            weight_scale.dtype()
        )
    }
    let dense_bytes = out_dim
        .checked_mul(in_dim)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| candle::Error::Msg("nvfp4: dense weight size overflows usize".into()))?;
    if dense_bytes > max_dense_bytes {
        candle::bail!(
            "nvfp4: dequantizing a [{out_dim}, {in_dim}] weight would allocate {dense_bytes} bytes, \
             exceeding the {max_dense_bytes}-byte budget; raise Nvfp4Config::max_dense_bytes or use \
             the nvfp4-cuda fused path instead"
        )
    }

    let packed = packed_weight.to_vec2::<u8>()?;
    let scale = weight_scale.to_vec2::<float8::F8E4M3>()?;

    let mut out = vec![0f32; out_dim * in_dim];
    for row in 0..out_dim {
        let packed_row = &packed[row];
        let scale_row = &scale[row];
        for col in 0..in_dim {
            let byte = packed_row[col / 2];
            let nibble = if col.is_multiple_of(2) {
                byte & 0xf
            } else {
                byte >> 4
            };
            let block_scale = scale_row[col / block_size].to_f32();
            out[row * in_dim + col] = unpack_e2m1(nibble) * block_scale * weight_scale_2;
        }
    }
    Tensor::from_vec(out, (out_dim, in_dim), &candle::Device::Cpu)
}

/// Configuration for [`nvfp4_linear`], [`nvfp4_linear_packed`] and
/// [`auto_linear`].
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4Config {
    pub block_size: usize,
    /// Refuses to dequantize a packed weight into a dense tensor larger than
    /// this many bytes. Only applies to the CPU dequantization fallback; the
    /// `nvfp4-cuda` fused path and the CPU packed path never expand the
    /// weight.
    pub max_dense_bytes: usize,
    /// When [`auto_linear`] can't use the `nvfp4-cuda` fused path, it
    /// defaults to [`Nvfp4LinearPacked`] (packed, ~1x footprint). Set this to
    /// dequantize to a dense tensor once at load time instead
    /// ([`nvfp4_linear`]), trading resident memory for forward-pass
    /// throughput; still bounded by `max_dense_bytes`.
    pub force_dense_fallback: bool,
}

impl Default for Nvfp4Config {
    fn default() -> Self {
        Self {
            block_size: BLOCK_SIZE,
            // A generous but explicit guardrail: refuse rather than silently
            // blow past available memory on an 8x unpack.
            max_dense_bytes: 4 * 1024 * 1024 * 1024,
            force_dense_fallback: false,
        }
    }
}

/// A CPU operator for a packed NVFP4 linear layer that never materializes a
/// dense `[out_dim, in_dim]` weight: the packed weight and block scales stay
/// in their checkpoint-native, ~1x-footprint form, and each output row is
/// decoded on the fly during [`forward`](Module::forward). See the module
/// docs for where this sits in [`auto_linear`]'s execution preference order.
///
/// Unlike [`nvfp4_linear`], `forward` here ignores the `VarBuilder`'s dtype
/// entirely: decoding and the matmul both run in `f32`, and only the output
/// is rounded to the input's dtype at the very end. [`nvfp4_linear`]
/// instead rounds the dense weight to the `VarBuilder`'s dtype (e.g. `bf16`)
/// once at load time, so every subsequent matmul runs at that lower
/// precision. On a reduced-precision checkpoint the two paths therefore
/// aren't bit-identical -- this one is the more accurate of the two, since
/// it defers rounding instead of compounding it into the weight -- see
/// `nvfp4_linear_packed_precision_differs_from_dense_at_reduced_precision`.
#[derive(Debug, Clone)]
pub struct Nvfp4LinearPacked {
    /// Row-major `[out_dim, in_dim / 2]`, two E2M1 nibbles per byte.
    packed_weight: Vec<u8>,
    /// Row-major `[out_dim, in_dim.div_ceil(block_size)]`, kept as raw
    /// `F8E4M3` bytes (not decoded to `f32`) so this struct's footprint
    /// matches the checkpoint's, not 4x it; decoded per block in
    /// `forward`.
    weight_scale: Vec<float8::F8E4M3>,
    weight_scale_2: f32,
    in_dim: usize,
    out_dim: usize,
    block_size: usize,
    bias: Option<Tensor>,
}

/// Builds a [`Nvfp4LinearPacked`] from a packed NVFP4 checkpoint, reading the
/// same tensors as [`nvfp4_linear`] (packed weight, `weight_scale`,
/// `weight_scale_2`, optional `bias`) but keeping them packed instead of
/// dequantizing to a dense tensor.
pub fn nvfp4_linear_packed(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    cfg: Nvfp4Config,
) -> Result<Nvfp4LinearPacked> {
    let packed_weight = vb
        .get_with_hints_dtype(
            (out_dim, in_dim / 2),
            nvfp4_weight_tensor_name(&vb),
            Default::default(),
            DType::U8,
        )?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let weight_scale = vb
        .get_with_hints_dtype(
            (out_dim, in_dim.div_ceil(cfg.block_size)),
            "weight_scale",
            Default::default(),
            DType::F8E4M3,
        )?
        .flatten_all()?
        .to_vec1::<float8::F8E4M3>()?;
    let weight_scale_2 = read_scalar_f32(&vb, "weight_scale_2")?;
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    Ok(Nvfp4LinearPacked {
        packed_weight,
        weight_scale,
        weight_scale_2,
        in_dim,
        out_dim,
        block_size: cfg.block_size,
        bias,
    })
}

impl Module for Nvfp4LinearPacked {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let target_dtype = xs.dtype();
        let target_device = xs.device().clone();
        let dims = xs.dims();
        let Some((&last, leading)) = dims.split_last() else {
            candle::bail!("nvfp4: input tensor must have at least one dimension")
        };
        if last != self.in_dim {
            candle::bail!(
                "nvfp4: expected last input dim {}, got {:?}",
                self.in_dim,
                dims
            )
        }
        let batch: usize = leading.iter().product();

        let xs_host = xs
            .reshape((batch, self.in_dim))?
            .to_device(&candle::Device::Cpu)?
            .to_dtype(DType::F32)?;
        let xs_vec = xs_host.to_vec2::<f32>()?;

        let packed_row_len = self.in_dim / 2;
        let n_blocks = self.in_dim.div_ceil(self.block_size);
        let mut out = vec![0f32; batch * self.out_dim];
        let mut row_buf = vec![0f32; self.in_dim];
        for (o, (packed_row, scale_row)) in self
            .packed_weight
            .chunks(packed_row_len)
            .zip(self.weight_scale.chunks(n_blocks))
            .enumerate()
        {
            for (col, val) in row_buf.iter_mut().enumerate() {
                let byte = packed_row[col / 2];
                let nibble = if col.is_multiple_of(2) {
                    byte & 0xf
                } else {
                    byte >> 4
                };
                let block_scale = scale_row[col / self.block_size].to_f32();
                *val = unpack_e2m1(nibble) * block_scale * self.weight_scale_2;
            }
            for (b, x_row) in xs_vec.iter().enumerate() {
                let acc: f32 = row_buf.iter().zip(x_row.iter()).map(|(w, x)| w * x).sum();
                out[b * self.out_dim + o] = acc;
            }
        }

        let mut out_shape = leading.to_vec();
        out_shape.push(self.out_dim);
        let out_tensor = Tensor::from_vec(out, (batch, self.out_dim), &candle::Device::Cpu)?
            .reshape(out_shape)?
            .to_device(&target_device)?
            .to_dtype(target_dtype)?;
        match &self.bias {
            None => Ok(out_tensor),
            Some(bias) => out_tensor.broadcast_add(bias),
        }
    }
}

/// Builds a dense [`candle_nn::Linear`] layer from a packed NVFP4 checkpoint,
/// reading the packed weight ([`nvfp4_weight_tensor_name`], `U8`),
/// `weight_scale` (`F8E4M3`), `weight_scale_2` (one-element `f32`) and an
/// optional `bias` tensor at the current `VarBuilder` path, dequantizing on
/// the CPU.
pub fn nvfp4_linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    cfg: Nvfp4Config,
) -> Result<Linear> {
    let packed_weight = vb.get_with_hints_dtype(
        (out_dim, in_dim / 2),
        nvfp4_weight_tensor_name(&vb),
        Default::default(),
        DType::U8,
    )?;
    let weight_scale = vb.get_with_hints_dtype(
        (out_dim, in_dim.div_ceil(cfg.block_size)),
        "weight_scale",
        Default::default(),
        DType::F8E4M3,
    )?;
    let weight_scale_2 = read_scalar_f32(&vb, "weight_scale_2")?;
    let weight = dequantize_nvfp4(
        &packed_weight,
        &weight_scale,
        weight_scale_2,
        cfg.block_size,
        cfg.max_dense_bytes,
    )?
    .to_dtype(vb.dtype())?
    .to_device(vb.device())?;
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

/// A linear projection whose backing (dense, block-wise FP8, or packed
/// NVFP4) was determined automatically from the checkpoint. See
/// [`auto_linear`].
///
/// `#[non_exhaustive]`: a Metal-backed NVFP4 variant is expected future work
/// (huggingface/candle#3857), and this should not be a breaking change for
/// downstream matches when it lands.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AutoLinear {
    Dense(Linear),
    Fp8Block(Linear),
    /// Dense NVFP4 fallback: dequantized once at load time. Only produced
    /// when [`Nvfp4Config::force_dense_fallback`] is set; see
    /// [`Nvfp4Packed`](Self::Nvfp4Packed) for the default off-CUDA path.
    Nvfp4(Linear),
    /// Packed NVFP4 CPU operator: the default off-CUDA path. See
    /// [`Nvfp4LinearPacked`].
    Nvfp4Packed(Nvfp4LinearPacked),
    #[cfg(feature = "nvfp4-cuda")]
    Nvfp4Cuda(cuda::Nvfp4LinearCuda),
}

impl Module for AutoLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) | Self::Fp8Block(l) | Self::Nvfp4(l) => l.forward(xs),
            Self::Nvfp4Packed(l) => l.forward(xs),
            #[cfg(feature = "nvfp4-cuda")]
            Self::Nvfp4Cuda(l) => l.forward(xs),
        }
    }
}

/// A ModelOpt-aware weight source: detects what backs the linear layer at
/// `vb`'s current path (dense, block-wise FP8, or packed NVFP4) and builds a
/// compatible projection, so callers stop needing to special-case each
/// checkpoint's tensor layout themselves.
///
/// On a CUDA device with the `nvfp4-cuda` feature enabled and the fused
/// kernel available, NVFP4 layers stay packed and device-resident. Otherwise
/// they default to [`Nvfp4LinearPacked`], which keeps the weight packed
/// (~1x footprint) and decodes on the fly; set
/// `cfg.force_dense_fallback` to dequantize to a dense tensor at load time
/// instead, bounded by `cfg.max_dense_bytes`. FP8 layers are always
/// dequantized to a dense tensor at load time. See the module docs for the
/// full preference order.
pub fn auto_linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    cfg: Nvfp4Config,
) -> Result<AutoLinear> {
    match detect_backing(&vb) {
        Backing::Nvfp4 => {
            #[cfg(feature = "nvfp4-cuda")]
            if vb.device().is_cuda() && candle_nvfp4_kernels::is_available() {
                return Ok(AutoLinear::Nvfp4Cuda(cuda::Nvfp4LinearCuda::new(
                    in_dim, out_dim, bias, vb, cfg,
                )?));
            }
            if cfg.force_dense_fallback {
                return Ok(AutoLinear::Nvfp4(nvfp4_linear(
                    in_dim, out_dim, bias, vb, cfg,
                )?));
            }
            Ok(AutoLinear::Nvfp4Packed(nvfp4_linear_packed(
                in_dim, out_dim, bias, vb, cfg,
            )?))
        }
        Backing::Fp8Block => Ok(AutoLinear::Fp8Block(fp8_block_linear(
            in_dim,
            out_dim,
            bias,
            vb,
            cfg.block_size,
        )?)),
        Backing::Dense => Ok(AutoLinear::Dense(candle_nn::linear_b(
            in_dim, out_dim, bias, vb,
        )?)),
    }
}

/// Fused path: keeps the NVFP4 weight packed and device-resident, running
/// the kernel from `candle-nvfp4-kernels` on every forward pass instead of
/// dequantizing once at load time.
#[cfg(feature = "nvfp4-cuda")]
pub mod cuda {
    use super::{nvfp4_weight_tensor_name, read_scalar_f32, Nvfp4Config};
    use candle::{DType, Module, Result, Tensor};
    use candle_nn::VarBuilder;

    #[derive(Debug, Clone)]
    pub struct Nvfp4LinearCuda {
        inner: candle_nvfp4_kernels::Nvfp4Linear,
        bias: Option<Tensor>,
    }

    impl Nvfp4LinearCuda {
        pub fn new(
            in_dim: usize,
            out_dim: usize,
            bias: bool,
            vb: VarBuilder,
            cfg: Nvfp4Config,
        ) -> Result<Self> {
            let packed_weight = vb
                .get_with_hints_dtype(
                    (out_dim, in_dim / 2),
                    nvfp4_weight_tensor_name(&vb),
                    Default::default(),
                    DType::U8,
                )?
                .flatten_all()?
                .to_vec1::<u8>()?;
            let weight_scale = vb
                .get_with_hints_dtype(
                    (out_dim, in_dim.div_ceil(cfg.block_size)),
                    "weight_scale",
                    Default::default(),
                    DType::F8E4M3,
                )?
                .flatten_all()?
                .to_vec1::<float8::F8E4M3>()?
                .into_iter()
                .map(|v| v.to_bits())
                .collect::<Vec<u8>>();
            let weight_scale_2 = read_scalar_f32(&vb, "weight_scale_2")?;
            let inner = candle_nvfp4_kernels::Nvfp4Linear::new(
                &packed_weight,
                &weight_scale,
                weight_scale_2,
                out_dim,
                in_dim,
                vb.device(),
            )?;
            let bias = if bias {
                Some(vb.get(out_dim, "bias")?)
            } else {
                None
            };
            Ok(Self { inner, bias })
        }
    }

    impl Module for Nvfp4LinearCuda {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let ys = self.inner.forward(xs)?;
            match &self.bias {
                None => Ok(ys),
                Some(bias) => ys.broadcast_add(bias),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn pack_row(values: &[f32]) -> Vec<u8> {
        // Reverse-maps a value to its nearest E2M1 code (test inputs are
        // chosen to be exact codepoints, so nearest is exact).
        const CODES: [f32; 16] = [
            0., 0.5, 1., 1.5, 2., 3., 4., 6., 0., -0.5, -1., -1.5, -2., -3., -4., -6.,
        ];
        let nibbles: Vec<u8> = values
            .iter()
            .map(|v| {
                CODES
                    .iter()
                    .position(|c| c == v)
                    .unwrap_or_else(|| panic!("{v} is not an exact E2M1 codepoint"))
                    as u8
            })
            .collect();
        nibbles
            .chunks(2)
            .map(|pair| pair[0] | (pair[1] << 4))
            .collect()
    }

    #[test]
    fn dequantize_nvfp4_matches_manual_computation() -> Result<()> {
        let device = Device::Cpu;
        let out_dim = 2;
        let in_dim = 32; // two blocks of 16
        let row0: Vec<f32> = vec![
            0., 0.5, 1., 1.5, 2., 3., 4., 6., -0.5, -1., -1.5, -2., -3., -4., -6., 0.,
        ];

        let packed_row = pack_row(&row0);
        let packed_bytes: Vec<u8> = [
            packed_row.clone(),
            packed_row.clone(),
            packed_row.clone(),
            packed_row,
        ]
        .concat();
        let packed_weight = Tensor::from_vec(packed_bytes, (out_dim, in_dim / 2), &device)?;

        // 1.0, 2.0 per row, built from raw E4M3 bit patterns (not a numeric
        // cast, which would round 0x38/0x40 as if they were integers).
        let scale_bytes: Vec<float8::F8E4M3> = [0x38u8, 0x40u8, 0x38u8, 0x40u8]
            .into_iter()
            .map(float8::F8E4M3::from_bits)
            .collect();
        let weight_scale = Tensor::from_vec(scale_bytes, (out_dim, in_dim / BLOCK_SIZE), &device)?;

        let tensor_scale = 0.5f32;
        let dequant = dequantize_nvfp4(
            &packed_weight,
            &weight_scale,
            tensor_scale,
            BLOCK_SIZE,
            1 << 30,
        )?;
        let dequant = dequant.to_vec2::<f32>()?;

        for (row, dequant_row) in dequant.iter().enumerate().take(out_dim) {
            for col in 0..in_dim {
                let block_scale = if col < BLOCK_SIZE { 1.0 } else { 2.0 };
                let expected = row0[col % BLOCK_SIZE] * block_scale * tensor_scale;
                assert!(
                    (dequant_row[col] - expected).abs() < 1e-5,
                    "mismatch at ({row},{col}): {} vs {expected}",
                    dequant_row[col]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn dequantize_nvfp4_refuses_over_budget_allocation() {
        let device = Device::Cpu;
        let out_dim = 4096;
        let in_dim = 4096;
        let packed_weight = Tensor::zeros((out_dim, in_dim / 2), DType::U8, &device).unwrap();
        let weight_scale =
            Tensor::zeros((out_dim, in_dim / BLOCK_SIZE), DType::F8E4M3, &device).unwrap();
        let err =
            dequantize_nvfp4(&packed_weight, &weight_scale, 1.0, BLOCK_SIZE, 1024).unwrap_err();
        assert!(err.to_string().contains("budget"));
    }

    #[test]
    fn detect_backing_reads_tensor_names() -> Result<()> {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        assert_eq!(detect_backing(&vb.pp("dense")), Backing::Dense);

        // ModelOpt convention: the packed weight is `.weight`, there is no
        // `.weight_packed` (e.g. nvidia/Llama-3.3-70B-Instruct-FP4).
        let _ = vb.pp("modelopt").get_with_hints_dtype(
            (4, 2),
            "weight",
            Default::default(),
            DType::U8,
        )?;
        let _ = vb.pp("modelopt").get_with_hints_dtype(
            (),
            "weight_scale_2",
            Default::default(),
            DType::F32,
        )?;
        assert_eq!(detect_backing(&vb.pp("modelopt")), Backing::Nvfp4);
        assert_eq!(nvfp4_weight_tensor_name(&vb.pp("modelopt")), "weight");

        // compressed-tensors convention (llm-compressor / vLLM): the packed
        // weight is `.weight_packed`, alongside `.weight_scale_2`.
        let _ = vb.pp("compressed_tensors").get_with_hints_dtype(
            (4, 2),
            "weight_packed",
            Default::default(),
            DType::U8,
        )?;
        let _ = vb.pp("compressed_tensors").get_with_hints_dtype(
            (),
            "weight_scale_2",
            Default::default(),
            DType::F32,
        )?;
        assert_eq!(detect_backing(&vb.pp("compressed_tensors")), Backing::Nvfp4);
        assert_eq!(
            nvfp4_weight_tensor_name(&vb.pp("compressed_tensors")),
            "weight_packed"
        );

        let _ = vb.pp("fp8").get_with_hints_dtype(
            (2, 2),
            "weight_scale_inv",
            Default::default(),
            DType::F32,
        )?;
        assert_eq!(detect_backing(&vb.pp("fp8")), Backing::Fp8Block);
        Ok(())
    }

    #[test]
    fn read_scalar_f32_accepts_rank_0_or_rank_1() -> Result<()> {
        let device = Device::Cpu;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "rank0.s".to_string(),
            Tensor::from_vec(vec![0.25f32], (), &device)?,
        );
        tensors.insert(
            "rank1.s".to_string(),
            Tensor::from_vec(vec![0.25f32], 1, &device)?,
        );
        tensors.insert(
            "not_scalar.s".to_string(),
            Tensor::from_vec(vec![0.25f32, 0.5], 2, &device)?,
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        assert_eq!(read_scalar_f32(&vb.pp("rank0"), "s")?, 0.25);
        assert_eq!(read_scalar_f32(&vb.pp("rank1"), "s")?, 0.25);
        assert!(read_scalar_f32(&vb.pp("not_scalar"), "s").is_err());
        Ok(())
    }

    #[test]
    fn nvfp4_linear_reads_the_modelopt_naming_convention() -> Result<()> {
        // A layer using ModelOpt's naming (packed weight under `.weight`,
        // no `.weight_packed`) must load through nvfp4_linear the same way
        // as the compressed-tensors naming; this is what nvfp4_weight_tensor_name
        // promises once detect_backing returns Backing::Nvfp4. See
        // auto_linear_dispatches_by_detected_backing for the auto_linear
        // entry point built on top of this.
        let device = Device::Cpu;
        let out_dim = 2;
        let in_dim = 32;
        let row0: Vec<f32> = vec![
            0., 0.5, 1., 1.5, 2., 3., 4., 6., -0.5, -1., -1.5, -2., -3., -4., -6., 0.,
        ];
        let packed_row = pack_row(&row0);
        let packed_bytes: Vec<u8> = [
            packed_row.clone(),
            packed_row.clone(),
            packed_row.clone(),
            packed_row,
        ]
        .concat();
        let scale_bytes: Vec<float8::F8E4M3> = [0x38u8, 0x38u8, 0x38u8, 0x38u8]
            .into_iter()
            .map(float8::F8E4M3::from_bits)
            .collect();

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "layer.weight".to_string(),
            Tensor::from_vec(packed_bytes, (out_dim, in_dim / 2), &device)?,
        );
        tensors.insert(
            "layer.weight_scale".to_string(),
            Tensor::from_vec(scale_bytes, (out_dim, in_dim / BLOCK_SIZE), &device)?,
        );
        tensors.insert(
            "layer.weight_scale_2".to_string(),
            Tensor::from_vec(vec![0.5f32], (), &device)?,
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let layer = vb.pp("layer");

        assert_eq!(detect_backing(&layer), Backing::Nvfp4);
        assert_eq!(nvfp4_weight_tensor_name(&layer), "weight");
        let linear = nvfp4_linear(in_dim, out_dim, false, layer, Nvfp4Config::default())?;
        let output = linear.forward(&Tensor::zeros((1, in_dim), DType::F32, &device)?)?;
        assert_eq!(output.dims(), &[1, out_dim]);
        Ok(())
    }

    #[test]
    fn auto_linear_dispatches_by_detected_backing() -> Result<()> {
        let device = Device::Cpu;
        let in_dim = 4;
        let out_dim = 2;
        let mut tensors = std::collections::HashMap::new();

        // Dense: a plain `.weight`, nothing else.
        tensors.insert(
            "dense.weight".to_string(),
            Tensor::zeros((out_dim, in_dim), DType::F32, &device)?,
        );

        // Block-wise FP8 (DeepSeek-style): `.weight` + `.weight_scale_inv`.
        tensors.insert(
            "fp8.weight".to_string(),
            Tensor::zeros((out_dim, in_dim), DType::F8E4M3, &device)?,
        );
        tensors.insert(
            "fp8.weight_scale_inv".to_string(),
            Tensor::from_vec(vec![1f32], (1, 1), &device)?,
        );

        // NVFP4, ModelOpt naming: `.weight` (packed) + `.weight_scale` +
        // `.weight_scale_2`, no `.weight_packed`. `row` must have exactly
        // `in_dim` values: it's one row of the packed weight, not a full
        // BLOCK_SIZE-wide block (in_dim=4 here is smaller than BLOCK_SIZE).
        let row: Vec<f32> = vec![0., 0.5, 1., 1.5];
        let packed_row = pack_row(&row);
        tensors.insert(
            "nvfp4.weight".to_string(),
            Tensor::from_vec(
                [packed_row.clone(), packed_row].concat(),
                (out_dim, in_dim / 2),
                &device,
            )?,
        );
        tensors.insert(
            "nvfp4.weight_scale".to_string(),
            Tensor::from_vec(
                vec![float8::F8E4M3::from_bits(0x38); out_dim],
                (out_dim, in_dim.div_ceil(BLOCK_SIZE)),
                &device,
            )?,
        );
        tensors.insert(
            "nvfp4.weight_scale_2".to_string(),
            Tensor::from_vec(vec![1f32], (), &device)?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let input = Tensor::zeros((1, in_dim), DType::F32, &device)?;

        let dense = auto_linear(
            in_dim,
            out_dim,
            false,
            vb.pp("dense"),
            Nvfp4Config::default(),
        )?;
        assert!(matches!(dense, AutoLinear::Dense(_)));
        assert_eq!(dense.forward(&input)?.dims(), &[1, out_dim]);

        let fp8 = auto_linear(in_dim, out_dim, false, vb.pp("fp8"), Nvfp4Config::default())?;
        assert!(matches!(fp8, AutoLinear::Fp8Block(_)));
        assert_eq!(fp8.forward(&input)?.dims(), &[1, out_dim]);

        let nvfp4 = auto_linear(
            in_dim,
            out_dim,
            false,
            vb.pp("nvfp4"),
            Nvfp4Config::default(),
        )?;
        assert!(matches!(nvfp4, AutoLinear::Nvfp4Packed(_)));
        assert_eq!(nvfp4.forward(&input)?.dims(), &[1, out_dim]);

        let nvfp4_dense = auto_linear(
            in_dim,
            out_dim,
            false,
            vb.pp("nvfp4"),
            Nvfp4Config {
                force_dense_fallback: true,
                ..Nvfp4Config::default()
            },
        )?;
        assert!(matches!(nvfp4_dense, AutoLinear::Nvfp4(_)));
        assert_eq!(nvfp4_dense.forward(&input)?.dims(), &[1, out_dim]);

        Ok(())
    }

    #[test]
    fn nvfp4_linear_packed_matches_dense_dequantization() -> Result<()> {
        let device = Device::Cpu;
        let out_dim = 2;
        let in_dim = 32; // two blocks of 16
        let row0: Vec<f32> = vec![
            0., 0.5, 1., 1.5, 2., 3., 4., 6., -0.5, -1., -1.5, -2., -3., -4., -6., 0.,
        ];
        let packed_row = pack_row(&row0);
        let packed_bytes: Vec<u8> = [
            packed_row.clone(),
            packed_row.clone(),
            packed_row.clone(),
            packed_row,
        ]
        .concat();
        let scale_bytes: Vec<float8::F8E4M3> = [0x38u8, 0x40u8, 0x38u8, 0x40u8]
            .into_iter()
            .map(float8::F8E4M3::from_bits)
            .collect();

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "layer.weight".to_string(),
            Tensor::from_vec(packed_bytes, (out_dim, in_dim / 2), &device)?,
        );
        tensors.insert(
            "layer.weight_scale".to_string(),
            Tensor::from_vec(scale_bytes, (out_dim, in_dim / BLOCK_SIZE), &device)?,
        );
        tensors.insert(
            "layer.weight_scale_2".to_string(),
            Tensor::from_vec(vec![0.5f32], (), &device)?,
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let dense = nvfp4_linear(
            in_dim,
            out_dim,
            false,
            vb.pp("layer"),
            Nvfp4Config::default(),
        )?;
        let packed = nvfp4_linear_packed(
            in_dim,
            out_dim,
            false,
            vb.pp("layer"),
            Nvfp4Config::default(),
        )?;

        // A batched, non-trivial input exercises the row/batch loops in
        // Nvfp4LinearPacked::forward, not just the zero vector.
        let xs = Tensor::rand(0f32, 1f32, (3, 5, in_dim), &device)?;
        let dense_out = dense.forward(&xs)?.to_vec3::<f32>()?;
        let packed_out = packed.forward(&xs)?.to_vec3::<f32>()?;

        for (d_batch, p_batch) in dense_out.iter().zip(packed_out.iter()) {
            for (d_row, p_row) in d_batch.iter().zip(p_batch.iter()) {
                for (d, p) in d_row.iter().zip(p_row.iter()) {
                    assert!((d - p).abs() < 1e-4, "{d} vs {p}");
                }
            }
        }
        Ok(())
    }

    /// Covers the precision note on [`Nvfp4LinearPacked`]'s docs: at a
    /// reduced-precision `VarBuilder` dtype, `nvfp4_linear` rounds the dense
    /// weight to that dtype once at load time (so rounding error compounds
    /// into every accumulation of the matmul), while `Nvfp4LinearPacked`
    /// always computes in `f32` and rounds only the final output. Neither is
    /// a bug, but they aren't bit-identical, and the `f32`-only parity test
    /// above doesn't exercise this at all.
    #[test]
    fn nvfp4_linear_packed_precision_differs_from_dense_at_reduced_precision() -> Result<()> {
        let device = Device::Cpu;
        let out_dim = 2;
        let in_dim = 32; // two blocks of 16
        let row0: Vec<f32> = vec![
            0., 0.5, 1., 1.5, 2., 3., 4., 6., -0.5, -1., -1.5, -2., -3., -4., -6., 0.,
        ];
        let packed_row = pack_row(&row0);
        let packed_bytes: Vec<u8> = [
            packed_row.clone(),
            packed_row.clone(),
            packed_row.clone(),
            packed_row,
        ]
        .concat();
        let scale_bytes: Vec<float8::F8E4M3> = [0x38u8, 0x40u8, 0x38u8, 0x40u8]
            .into_iter()
            .map(float8::F8E4M3::from_bits)
            .collect();

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "layer.weight".to_string(),
            Tensor::from_vec(packed_bytes, (out_dim, in_dim / 2), &device)?,
        );
        tensors.insert(
            "layer.weight_scale".to_string(),
            Tensor::from_vec(scale_bytes, (out_dim, in_dim / BLOCK_SIZE), &device)?,
        );
        // A non-power-of-two tensor scale: a power-of-two scale wouldn't
        // cost any f16 mantissa bits and would defeat the point of this
        // test, since E2M1 codepoints are already exact in f16.
        tensors.insert(
            "layer.weight_scale_2".to_string(),
            Tensor::from_vec(vec![1f32 / 3.], (), &device)?,
        );
        let vb_f32 = VarBuilder::from_tensors(tensors.clone(), DType::F32, &device);
        // candle-core's default CPU matmul only supports F16/F32/F64, not
        // BF16 -- nvfp4_linear's dense path would fail outright at BF16
        // here regardless of anything in this module, so this test uses
        // F16 to isolate the rounding-order question it's actually about.
        let vb_f16 = VarBuilder::from_tensors(tensors, DType::F16, &device);

        // Nvfp4LinearPacked never reads vb.dtype() (see nvfp4_linear_packed):
        // building it from an f32 VarBuilder is enough, the reduced
        // precision below comes entirely from the input/output dtype.
        let packed = nvfp4_linear_packed(
            in_dim,
            out_dim,
            false,
            vb_f32.pp("layer"),
            Nvfp4Config::default(),
        )?;
        // nvfp4_linear does read vb.dtype() and rounds the dense weight to
        // it, so this one needs the f16 VarBuilder.
        let dense_f16 = nvfp4_linear(
            in_dim,
            out_dim,
            false,
            vb_f16.pp("layer"),
            Nvfp4Config::default(),
        )?;

        // A deterministic, non-uniform input: Tensor::rand is unseeded, and
        // an unlucky draw could occasionally make packed's single rounding
        // step land as large as dense's compounded one, flaking the
        // dense_max_abs_diff > packed_max_abs_diff assertion below.
        let batch = 3 * 5;
        let xs_data: Vec<f32> = (0..batch * in_dim)
            .map(|i| 0.01 + 0.037 * ((i % 13) as f32))
            .collect();
        let xs_f16 = Tensor::from_vec(xs_data, (3, 5, in_dim), &device)?.to_dtype(DType::F16)?;
        // What Nvfp4LinearPacked::forward computes with internally: xs_f16
        // upconverted to f32, losslessly (f16 -> f32 never rounds).
        let xs_ref = xs_f16.to_dtype(DType::F32)?;

        // True f32 reference: same dequantized weight and input as both
        // paths compute from, no rounding anywhere.
        let reference = packed.forward(&xs_ref)?.to_vec3::<f32>()?;

        // Packed's only rounding step is the final output cast to f16, so
        // this must match `reference` rounded to f16 exactly.
        let packed_out = packed
            .forward(&xs_f16)?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;
        let rounded_reference = Tensor::new(reference.clone(), &device)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;
        for (p_batch, r_batch) in packed_out.iter().zip(rounded_reference.iter()) {
            for (p_row, r_row) in p_batch.iter().zip(r_batch.iter()) {
                for (p, r) in p_row.iter().zip(r_row.iter()) {
                    assert_eq!(
                        p, r,
                        "packed's f16 output should equal the f32 reference rounded once at the end"
                    );
                }
            }
        }

        // Dense rounds the weight to f16 before the matmul, so its error
        // compounds across all in_dim=32 accumulations instead of rounding
        // once at the end like the packed path does. Compare each against
        // the same f32 reference rather than asserting an absolute
        // threshold, since the right magnitude depends on the random input:
        // dense's compounded error must exceed packed's single
        // rounding-only error, not just be nonzero.
        let dense_out = dense_f16
            .forward(&xs_f16)?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;
        let mut dense_max_abs_diff = 0f32;
        let mut packed_max_abs_diff = 0f32;
        for ((d_batch, p_batch), r_batch) in dense_out
            .iter()
            .zip(packed_out.iter())
            .zip(reference.iter())
        {
            for ((d_row, p_row), r_row) in d_batch.iter().zip(p_batch.iter()).zip(r_batch.iter()) {
                for ((d, p), r) in d_row.iter().zip(p_row.iter()).zip(r_row.iter()) {
                    dense_max_abs_diff = dense_max_abs_diff.max((d - r).abs());
                    packed_max_abs_diff = packed_max_abs_diff.max((p - r).abs());
                }
            }
        }
        assert!(
            dense_max_abs_diff > packed_max_abs_diff,
            "expected dense's compounded f16 rounding ({dense_max_abs_diff}) to diverge \
             from the f32 reference more than packed's single final rounding \
             ({packed_max_abs_diff})"
        );
        Ok(())
    }
}
