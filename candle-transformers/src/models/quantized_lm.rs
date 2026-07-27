//! Architecture dispatch for quantized (GGUF) language models.
//!
//! `candle-transformers` ships one `quantized_*` module per model family, each with its
//! own constructor and its own metadata namespace. Loading "whatever GGUF this is" therefore
//! means writing a dispatcher on `general.architecture`, plus remembering that
//! `llama.context_length` and `qwen3.context_length` are different keys. This module does
//! both once:
//!
//! - [`QuantizedLm`] is the decoder interface every family already implements, so a loaded
//!   model can be used behind a `Box<dyn QuantizedLm>` without a wrapper enum.
//! - [`from_gguf`] reads `general.architecture` and constructs the matching backend,
//!   normalizing the per-family extras (`use_flash_attn`, working `dtype`) into [`Options`].
//! - [`context_length`] and [`arch_metadata`] read architecture-prefixed metadata through the
//!   file's own namespace, so a consumer cannot silently read `llama.*` out of a Qwen file.
//!
//! Unsupported `(architecture, device, dtype)` combinations are rejected before any tensor is
//! read — see [`Architecture::check_device_support`].
//!
//! ```no_run
//! use candle::{Device, quantized::gguf_file};
//! use candle_transformers::models::quantized_lm;
//!
//! # fn main() -> candle::Result<()> {
//! let device = Device::Cpu;
//! let mut file = std::fs::File::open("model.gguf")?;
//! let content = gguf_file::Content::read(&mut file)?;
//! let ctx = quantized_lm::context_length(&content);
//! let mut model = quantized_lm::from_gguf(content, &mut file, &device)?;
//! # let _ = (ctx, &mut model);
//! # Ok(())
//! # }
//! ```

use candle::quantized::gguf_file;
use candle::{DType, Device, Result, Tensor};
use std::io::{Read, Seek};

use super::{
    quantized_gemma3, quantized_glm4, quantized_lfm2, quantized_llama, quantized_phi,
    quantized_phi3, quantized_qwen2, quantized_qwen3, quantized_qwen3_moe,
};

/// The decoder interface shared by the quantized model families.
///
/// `index_pos` is the number of tokens already in the KV cache, i.e. the position of the first
/// token of `input` in the sequence.
pub trait QuantizedLm {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor>;
}

/// A model family with a quantized GGUF backend in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    /// [`quantized_llama`], `general.architecture = "llama"`.
    Llama,
    /// [`quantized_gemma3`], covers the whole gemma line.
    Gemma3,
    /// [`quantized_glm4`], `general.architecture = "glm4"`.
    Glm4,
    /// [`quantized_lfm2`], `general.architecture = "lfm2"`.
    Lfm2,
    /// [`quantized_phi`], `general.architecture = "phi2"`.
    Phi2,
    /// [`quantized_phi3`], `general.architecture = "phi3"`.
    Phi3,
    /// [`quantized_qwen2`], `general.architecture = "qwen2"`.
    Qwen2,
    /// [`quantized_qwen3`], `general.architecture = "qwen3"`.
    Qwen3,
    /// [`quantized_qwen3_moe`], `general.architecture = "qwen3moe"`. CUDA only.
    Qwen3Moe,
}

/// Every `general.architecture` value [`from_gguf`] accepts, in dispatch order.
pub const SUPPORTED_ARCHITECTURES: &[&str] = &[
    "llama",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma-embedding",
    "glm4",
    "lfm2",
    "phi2",
    "phi3",
    "qwen2",
    "qwen3",
    "qwen3moe",
];

impl Architecture {
    /// Map a `general.architecture` value onto the backend that handles it.
    ///
    /// The gemma backend reads several prefixes, so `gemma`, `gemma2`, `gemma3` and
    /// `gemma-embedding` all map to [`Architecture::Gemma3`].
    pub fn from_name(name: &str) -> Option<Self> {
        let arch = match name {
            "llama" => Self::Llama,
            "gemma" | "gemma2" | "gemma3" | "gemma-embedding" => Self::Gemma3,
            "glm4" => Self::Glm4,
            "lfm2" => Self::Lfm2,
            "phi2" => Self::Phi2,
            "phi3" => Self::Phi3,
            "qwen2" => Self::Qwen2,
            "qwen3" => Self::Qwen3,
            "qwen3moe" => Self::Qwen3Moe,
            _ => return None,
        };
        Some(arch)
    }

    /// Read `general.architecture` from `ct` and resolve it to a backend.
    pub fn from_content(ct: &gguf_file::Content) -> Result<Self> {
        let name = architecture_name(ct)?;
        match Self::from_name(name) {
            Some(arch) => Ok(arch),
            None => candle::bail!(
                "unsupported gguf architecture {name:?}, expected one of {SUPPORTED_ARCHITECTURES:?}"
            ),
        }
    }

    /// The canonical `general.architecture` value for this backend.
    ///
    /// This is not a round-trip of [`Architecture::from_name`] for the aliased families:
    /// a `gemma2` file resolves to [`Architecture::Gemma3`], whose name is `"gemma3"`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Gemma3 => "gemma3",
            Self::Glm4 => "glm4",
            Self::Lfm2 => "lfm2",
            Self::Phi2 => "phi2",
            Self::Phi3 => "phi3",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3moe",
        }
    }

    /// Reject a `(device, dtype)` combination this backend cannot run on.
    ///
    /// [`from_gguf`] calls this before reading any tensor, so an unusable combination fails at
    /// load time rather than in the middle of the first forward pass.
    ///
    /// `qwen3moe` routes its experts through `candle_nn::moe_gemm_gguf`, which is implemented
    /// for CUDA only and whose prefill kernel takes an F16 or BF16 working dtype.
    pub fn check_device_support(&self, device: &Device, dtype: DType) -> Result<()> {
        if *self == Self::Qwen3Moe {
            if !device.is_cuda() {
                candle::bail!(
                    "the qwen3moe backend requires a cuda device: \
                     candle_nn::moe_gemm_gguf is only implemented for the cuda backend"
                )
            }
            if !matches!(dtype, DType::F16 | DType::BF16) {
                candle::bail!(
                    "the qwen3moe backend requires a f16 or bf16 working dtype, got {dtype:?}"
                )
            }
        }
        Ok(())
    }
}

/// Per-family construction extras, with defaults that suit every backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    /// Use flash-attention where the backend supports it. Only `phi3` reads this today.
    pub use_flash_attn: bool,
    /// Working dtype for the backends that take one (`glm4`, `qwen3moe`).
    ///
    /// `None` resolves to BF16 on CUDA and Metal, F32 elsewhere. Note that `qwen3moe` rejects
    /// F32, so a CPU load of that family fails on the device check first either way.
    pub dtype: Option<DType>,
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.use_flash_attn = use_flash_attn;
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// The working dtype to use on `device`, resolving the `None` default.
    pub fn dtype_for(&self, device: &Device) -> DType {
        self.dtype.unwrap_or({
            if device.is_cuda() || device.is_metal() {
                DType::BF16
            } else {
                DType::F32
            }
        })
    }
}

/// The raw `general.architecture` value, e.g. `"qwen3"`.
pub fn architecture_name(ct: &gguf_file::Content) -> Result<&str> {
    match ct.metadata.get("general.architecture") {
        None => candle::bail!("cannot find general.architecture in metadata"),
        Some(v) => Ok(v.to_string()?.as_str()),
    }
}

/// Read a metadata entry from the file's own architecture namespace.
///
/// `arch_metadata(ct, "context_length")` reads `qwen3.context_length` out of a Qwen3 file and
/// `llama.context_length` out of a llama one, which is what consumers hardcoding a single
/// prefix get wrong. Returns `None` when `general.architecture` or the entry is missing.
pub fn arch_metadata<'a>(ct: &'a gguf_file::Content, suffix: &str) -> Option<&'a gguf_file::Value> {
    let arch = architecture_name(ct).ok()?;
    ct.metadata.get(&format!("{arch}.{suffix}"))
}

/// The training context window of the checkpoint, read from `{architecture}.context_length`.
///
/// Returns `None` when the key is absent or not an integer — callers should fall back to their
/// own default rather than to another family's namespace.
pub fn context_length(ct: &gguf_file::Content) -> Option<usize> {
    let v = arch_metadata(ct, "context_length")?;
    v.to_u64().ok().map(|v| v as usize)
}

/// Load a GGUF checkpoint with the backend registered for its `general.architecture`.
///
/// Uses [`Options::default`]; see [`from_gguf_with`] to pass a working dtype or enable
/// flash-attention.
pub fn from_gguf<R: Read + Seek>(
    ct: gguf_file::Content,
    reader: &mut R,
    device: &Device,
) -> Result<Box<dyn QuantizedLm>> {
    from_gguf_with(ct, reader, device, &Options::default())
}

/// Load a GGUF checkpoint with the backend registered for its `general.architecture`.
///
/// Fails before reading any tensor when the architecture is unknown, or when the backend
/// cannot run on `device` with the resolved working dtype.
pub fn from_gguf_with<R: Read + Seek>(
    ct: gguf_file::Content,
    reader: &mut R,
    device: &Device,
    options: &Options,
) -> Result<Box<dyn QuantizedLm>> {
    let arch = Architecture::from_content(&ct)?;
    let dtype = options.dtype_for(device);
    arch.check_device_support(device, dtype)?;
    let model: Box<dyn QuantizedLm> = match arch {
        Architecture::Llama => Box::new(quantized_llama::ModelWeights::from_gguf(
            ct, reader, device,
        )?),
        Architecture::Gemma3 => Box::new(quantized_gemma3::ModelWeights::from_gguf(
            ct, reader, device,
        )?),
        Architecture::Glm4 => Box::new(quantized_glm4::ModelWeights::from_gguf(
            ct, reader, device, dtype,
        )?),
        Architecture::Lfm2 => {
            Box::new(quantized_lfm2::ModelWeights::from_gguf(ct, reader, device)?)
        }
        Architecture::Phi2 => Box::new(quantized_phi::ModelWeights::from_gguf(ct, reader, device)?),
        Architecture::Phi3 => Box::new(quantized_phi3::ModelWeights::from_gguf(
            options.use_flash_attn,
            ct,
            reader,
            device,
        )?),
        Architecture::Qwen2 => Box::new(quantized_qwen2::ModelWeights::from_gguf(
            ct, reader, device,
        )?),
        Architecture::Qwen3 => Box::new(quantized_qwen3::ModelWeights::from_gguf(
            ct, reader, device,
        )?),
        Architecture::Qwen3Moe => Box::new(quantized_qwen3_moe::GGUFQWenMoE::from_gguf(
            ct, reader, device, dtype,
        )?),
    };
    Ok(model)
}

macro_rules! impl_quantized_lm {
    ($ty:ty) => {
        impl QuantizedLm for $ty {
            fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
                <$ty>::forward(self, input, index_pos)
            }
        }
    };
}

impl_quantized_lm!(quantized_llama::ModelWeights);
impl_quantized_lm!(quantized_gemma3::ModelWeights);
impl_quantized_lm!(quantized_glm4::ModelWeights);
impl_quantized_lm!(quantized_lfm2::ModelWeights);
impl_quantized_lm!(quantized_phi::ModelWeights);
impl_quantized_lm!(quantized_phi3::ModelWeights);
impl_quantized_lm!(quantized_qwen2::ModelWeights);
impl_quantized_lm!(quantized_qwen3::ModelWeights);
impl_quantized_lm!(quantized_qwen3_moe::GGUFQWenMoE);

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::gguf_file::{Content, Value, VersionedMagic};
    use std::collections::HashMap;
    use std::io::Cursor;

    fn content(entries: &[(&str, Value)]) -> Content {
        let metadata = entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect::<HashMap<_, _>>();
        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        }
    }

    #[test]
    fn architecture_names_resolve() {
        for name in SUPPORTED_ARCHITECTURES {
            assert!(
                Architecture::from_name(name).is_some(),
                "{name} is advertised but does not dispatch"
            );
        }
        assert_eq!(Architecture::from_name("qwen3"), Some(Architecture::Qwen3));
        assert_eq!(
            Architecture::from_name("qwen3moe"),
            Some(Architecture::Qwen3Moe)
        );
        // The gemma backend probes several prefixes, they share one entry point.
        for name in ["gemma", "gemma2", "gemma3", "gemma-embedding"] {
            assert_eq!(Architecture::from_name(name), Some(Architecture::Gemma3));
        }
        assert_eq!(Architecture::from_name("qwen"), None);
        assert_eq!(Architecture::from_name("mamba"), None);
        assert_eq!(Architecture::from_name(""), None);
    }

    #[test]
    fn canonical_names_dispatch_back_to_themselves() {
        for arch in [
            Architecture::Llama,
            Architecture::Gemma3,
            Architecture::Glm4,
            Architecture::Lfm2,
            Architecture::Phi2,
            Architecture::Phi3,
            Architecture::Qwen2,
            Architecture::Qwen3,
            Architecture::Qwen3Moe,
        ] {
            assert_eq!(Architecture::from_name(arch.name()), Some(arch));
            assert!(SUPPORTED_ARCHITECTURES.contains(&arch.name()));
        }
    }

    #[test]
    fn missing_architecture_is_an_error() {
        let ct = content(&[("qwen3.context_length", Value::U32(40960))]);
        assert!(architecture_name(&ct).is_err());
        assert!(Architecture::from_content(&ct).is_err());
        assert_eq!(context_length(&ct), None);
    }

    #[test]
    fn unknown_architecture_lists_the_supported_ones() {
        let ct = content(&[("general.architecture", Value::String("mamba".to_string()))]);
        let err = Architecture::from_content(&ct).unwrap_err().to_string();
        assert!(err.contains("mamba"), "{err}");
        assert!(err.contains("qwen3moe"), "{err}");
    }

    #[test]
    fn context_length_follows_the_files_own_namespace() {
        let qwen = content(&[
            ("general.architecture", Value::String("qwen3".to_string())),
            ("qwen3.context_length", Value::U32(40960)),
            // A llama-prefixed key must not leak into a qwen file.
            ("llama.context_length", Value::U32(2048)),
        ]);
        assert_eq!(context_length(&qwen), Some(40960));

        let llama = content(&[
            ("general.architecture", Value::String("llama".to_string())),
            ("llama.context_length", Value::U32(2048)),
        ]);
        assert_eq!(context_length(&llama), Some(2048));

        // The footgun: a qwen file carrying no llama namespace yields None rather than a
        // silently wrong default.
        let bare = content(&[
            ("general.architecture", Value::String("qwen3".to_string())),
            ("llama.context_length", Value::U32(2048)),
        ]);
        assert_eq!(context_length(&bare), None);
    }

    #[test]
    fn context_length_upcasts_integer_widths() {
        let ct = content(&[
            ("general.architecture", Value::String("llama".to_string())),
            ("llama.context_length", Value::U64(131072)),
        ]);
        assert_eq!(context_length(&ct), Some(131072));

        let bad = content(&[
            ("general.architecture", Value::String("llama".to_string())),
            ("llama.context_length", Value::String("4096".to_string())),
        ]);
        assert_eq!(context_length(&bad), None);
    }

    #[test]
    fn arch_metadata_reads_arbitrary_suffixes() {
        let ct = content(&[
            ("general.architecture", Value::String("phi3".to_string())),
            ("phi3.block_count", Value::U32(32)),
        ]);
        assert_eq!(
            arch_metadata(&ct, "block_count").and_then(|v| v.to_u64().ok()),
            Some(32)
        );
        assert!(arch_metadata(&ct, "expert_count").is_none());
    }

    #[test]
    fn qwen3moe_is_rejected_off_cuda_before_any_read() {
        let ct = content(&[(
            "general.architecture",
            Value::String("qwen3moe".to_string()),
        )]);
        // An empty reader: the load must fail on the device check, not on a short read.
        let mut reader = Cursor::new(Vec::new());
        let err = match from_gguf(ct, &mut reader, &Device::Cpu) {
            Ok(_) => panic!("qwen3moe loaded on the cpu"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("cuda"), "{err}");
    }

    #[test]
    fn only_qwen3moe_constrains_the_device() {
        // Even the dtype qwen3moe wants is not enough off cuda.
        let err = Architecture::Qwen3Moe
            .check_device_support(&Device::Cpu, DType::F16)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cuda"), "{err}");
        // Every other family is device and dtype agnostic here.
        for arch in [Architecture::Llama, Architecture::Glm4, Architecture::Qwen3] {
            arch.check_device_support(&Device::Cpu, DType::F32).unwrap();
        }
    }

    #[test]
    fn default_options_resolve_a_cpu_dtype() {
        let options = Options::default();
        assert!(!options.use_flash_attn);
        assert_eq!(options.dtype, None);
        assert_eq!(options.dtype_for(&Device::Cpu), DType::F32);

        let options = Options::new().with_dtype(DType::BF16).with_flash_attn(true);
        assert!(options.use_flash_attn);
        assert_eq!(options.dtype_for(&Device::Cpu), DType::BF16);
    }
}
