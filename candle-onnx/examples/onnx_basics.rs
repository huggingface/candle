use anyhow::Result;
use prost::Message;

pub fn main() -> Result<()> {
    let buf = std::fs::read("test.onnx")?;
    let v = candle_onnx::onnx::ModelProto::decode(buf.as_slice())?;
    println!("{v:?}");
    Ok(())
}
