use candle::{DType, Result};
use std::path::Path;

pub fn detect_dtype<P: AsRef<Path>>(filenames: &[P]) -> Result<DType> {
    for filename in filenames {
        if filename.as_ref().extension().and_then(|s| s.to_str()) == Some("safetensors") {
            let file = std::fs::File::open(filename)?;
            let buffer = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let st = safetensors::SafeTensors::deserialize(&buffer)
                .map_err(|e| candle::Error::Msg(format!("safetensors error: {e}")))?;
            
            for (_name, tensor) in st.tensors() {
                return DType::try_from(tensor.dtype());
            }
        }
    }
    Ok(DType::F32) // Fallback
}
