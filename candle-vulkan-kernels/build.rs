use std::fs;
use std::path::Path;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let src_dir = Path::new("shaders");
    // Slang production shaders live next to the `.comp` files and are
    // compiled with `slangc`.
    // Re-run when the directory itself changes (e.g. a new shader file
    // appears): `rerun-if-changed` on individual files only tracks files
    // that existed at the last build.
    println!("cargo:rerun-if-changed={}", src_dir.display());
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
    println!("cargo:rerun-if-changed={}", test_src_dir.display());
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
    let mut args: Vec<String> = vec![
        source_path.to_string_lossy().into_owned(),
        "-o".to_string(),
        out_path.to_string_lossy().into_owned(),
        "-profile".to_string(),
        "glsl_460".to_string(),
        "-target".to_string(),
        "spirv".to_string(),
    ];
    // Extra slangc flags from the SLANGCFLAGS environment variable.
    if let Ok(extra) = std::env::var("SLANGCFLAGS") {
        for flag in extra.split_whitespace() {
            args.push(flag.to_string());
        }
    }
    let status = std::process::Command::new(&slangc)
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .status()
        .unwrap();

    if !status.success() {
        panic!("slangc failed compiling source '{source_path:?}'.");
    }
}
