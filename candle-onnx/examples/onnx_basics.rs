use anyhow::Result;

pub fn main() -> Result<()> {
    let v = candle_onnx::read_file("test.onnx")?;
    println!("{v:?}");
    Ok(())
}
