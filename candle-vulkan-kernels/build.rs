use std::fs;
use std::path::Path;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let src_dir = Path::new("shaders");
    // Slang production shaders live next to the `.comp` files and are
    // compiled with `slangc`.
    for entry in fs::read_dir(src_dir).expect("failed to read shaders dir") {
        let entry = entry.expect("failed to read shaders dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("slang") {
            continue;
        }
        let name = path
            .file_stem()
            .expect("slang shader without file stem")
            .to_string_lossy();
        let out_path = Path::new(&out_dir).join(format!("{name}.spv"));
        compile_slang(&path, &out_path);
        println!("cargo:rerun-if-changed={}", path.display());
    }
    // Test-only shaders live under `shaders/test/`. The SPIR-V output goes
    // to `${OUT_DIR}/test/` so a test shader can never collide with a
    // production shader of the same name.
    let test_src_dir = Path::new("shaders/test");
    let test_out_dir = Path::new(&out_dir).join("test");
    fs::create_dir_all(&test_out_dir).expect("failed to create test SPIR-V directory");
    for entry in fs::read_dir(test_src_dir).expect("failed to read shaders/test") {
        let entry = entry.expect("failed to read shaders/test entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("slang") {
            continue;
        }
        let name = path
            .file_stem()
            .expect("slang shader without file stem")
            .to_string_lossy();
        let out_path = test_out_dir.join(format!("{name}.spv"));
        compile_slang(&path, &out_path);
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn compile_slang(source_path: &Path, out_path: &Path) {
    // `SLANGC` overrides the slangc executable (e.g. a specific installation
    // or version); unset or empty falls back to `slangc` from the PATH.
    let slangc = match std::env::var("SLANGC") {
        Ok(value) if !value.is_empty() => value,
        _ => format!("slangc{}", if cfg!(windows) { ".exe" } else { "" }),
    };
    let status = std::process::Command::new(&slangc)
        .args(&[
            source_path.to_string_lossy().into_owned().as_str(),
            "-profile",
            "glsl_450",
            "-target",
            "spirv",
            "-o",
            out_path.to_string_lossy().into_owned().as_str(),
        ])
        .stdout(std::process::Stdio::piped())
        .status()
        .unwrap();

    if !status.success() {
        panic!("slangc failed compiling source '{source_path:?}'.");
    }
}
