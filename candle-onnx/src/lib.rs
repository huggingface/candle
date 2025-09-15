use candle::Result;
pub use prost;
use prost::Message;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod eval;
pub use eval::{dtype, simple_eval};

pub fn read_file<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto> {
    let buf = std::fs::read(p)?;
    onnx::ModelProto::decode(buf.as_slice()).map_err(candle::Error::wrap)
}
