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
        // Module files (a `module` declaration, no entry points of their
        // own) are pulled into the kernels that `import` them, so they are
        // never compiled standalone.
        if is_slang_module(&path) {
            println!("cargo:rerun-if-changed={}", path.display());
            continue;
        }
        let name = path
            .file_stem()
            .expect("slang shader without file stem")
            .to_string_lossy();
        let out_path = Path::new(&out_dir).join(format!("{name}.spv"));
        compile_slang(&path, &out_path, None);
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
        compile_slang(&path, &out_path, Some(src_dir));
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn compile_slang(source_path: &Path, out_path: &Path, include_dir: Option<&Path>) {
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
    // Include search path so `import` resolves module files that live in
    // `shaders/`; needed for the test shaders under `shaders/test/`.
    if let Some(dir) = include_dir {
        args.push("-I".to_string());
        args.push(dir.to_string_lossy().into_owned());
    }
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

/// True if `path` is the primary file of a Slang module: its first
/// non-blank statement (ignoring comments) is a `module` declaration.
fn is_slang_module(path: &Path) -> bool {
    let Ok(content) = fs::read_to_string(path) else {
        return false;
    };
    for line in strip_comments(&content).lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        return t.starts_with("module");
    }
    false
}

/// Replace comments with blanks (newlines preserved) so the first real
/// statement can be located without comment content interfering.
fn strip_comments(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut chars = content.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '/' if chars.peek() == Some(&'/') => {
                while let Some(&c2) = chars.peek() {
                    if c2 == '\n' {
                        break;
                    }
                    chars.next();
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                loop {
                    match chars.next() {
                        None => break,
                        Some('*') if chars.peek() == Some(&'/') => {
                            chars.next();
                            break;
                        }
                        Some('\n') => out.push('\n'),
                        _ => {}
                    }
                }
            }
            _ => out.push(c),
        }
    }
    out
}
