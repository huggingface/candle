use std::io::Result;

fn main() -> Result<()> {
    let protoc = protoc_bin_vendored::protoc_bin_path()
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    std::env::set_var("PROTOC", protoc);
    prost_build::compile_protos(&["src/onnx.proto3"], &["src/"])?;
    Ok(())
}
