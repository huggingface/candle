use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(&["src/onnx.proto3"], &["src/"])?;
    Ok(())
}
