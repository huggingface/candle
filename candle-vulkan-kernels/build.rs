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
    // The slangc optimization flag depends on these environment
    // variables (see `slang_opt_flag`); re-run the build script when
    // any of them changes.
    println!("cargo:rerun-if-env-changed=PROFILE");
    println!("cargo:rerun-if-env-changed=CANDLE_VULKAN_SLANG_OPT");
    println!("cargo:rerun-if-env-changed=CANDLE_VULKAN_SLANG_NO_OPT");
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
        let mut extra_flags: Vec<&str> = coopmat_flags(&name).to_vec();
        extra_flags.extend_from_slice(precise_fp_flags(&name));
        compile_slang(&path, &out_path, None, &[], &extra_flags);
        println!("cargo:rerun-if-changed={}", path.display());
        // Native-variant pass: recompile the bf16/f8e4m3 kernels with
        // `-DBF16_NATIVE=1` and/or `-DFP8_NATIVE=1` so they use the
        // device's native `BFloat16` type (VK_KHR_shader_bfloat16 +
        // shaderBFloat16Type) and/or native `FloatE4M3` type
        // (VK_EXT_shader_float8 + shaderFloat8) when the device supports
        // them; the default `<name>.spv` above is the emulated fallback.
        let mut native_defines: Vec<&str> = Vec::new();
        if NATIVE_BF16_SHADERS.iter().any(|s| *s == &*name) {
            native_defines.push("BF16_NATIVE=1");
        }
        if NATIVE_FP8_SHADERS.iter().any(|s| *s == &*name) {
            native_defines.push("FP8_NATIVE=1");
        }
        if !native_defines.is_empty() {
            let native_out = Path::new(&out_dir).join(format!("{name}_native.spv"));
            compile_slang(&path, &native_out, None, native_defines.as_slice(), &[]);
        }
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
        compile_slang(&path, &out_path, Some(src_dir), &[], &[]);
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// Shaders that get the native-`BFloat16` variant (see the loop above):
/// their stems plus `_native.spv` are emitted with `-DBF16_NATIVE=1`.
const NATIVE_BF16_SHADERS: &[&str] = &[
    "affine_bf16",
    "binary_bf16",
    "cmp_bf16",
    "const_set_bf16",
    "conv_bf16",
    "copy2d_bf16",
    "gather_idx_bf16",
    "reduce_bf16",
    "unary_bf16",
    "where_bf16",
    "to_dtype_bf16_f16",
    "to_dtype_bf16_f32",
    "to_dtype_bf16_f64",
    "to_dtype_bf16_i16",
    "to_dtype_bf16_i32",
    "to_dtype_bf16_i64",
    "to_dtype_bf16_u32",
    "to_dtype_bf16_u8",
    "to_dtype_f16_bf16",
    "to_dtype_f32_bf16",
    "to_dtype_f64_bf16",
    "to_dtype_i16_bf16",
    "to_dtype_i32_bf16",
    "to_dtype_i64_bf16",
    "to_dtype_u32_bf16",
    "to_dtype_u8_bf16",
];

/// Shaders that get the native-`FloatE4M3` variant (see the loop above):
/// their stems plus `_native.spv` are emitted with `-DFP8_NATIVE=1`. The
/// two cross `bf16<->f8e4m3` shaders are in both lists and get both
/// defines.
const NATIVE_FP8_SHADERS: &[&str] = &[
    "affine_f8e4m3",
    "binary_f8e4m3",
    "cmp_f8e4m3",
    "unary_f8e4m3",
    "where_f8e4m3",
    "to_dtype_f8e4m3_f16",
    "to_dtype_f8e4m3_f32",
    "to_dtype_f8e4m3_f64",
    "to_dtype_f8e4m3_i16",
    "to_dtype_f8e4m3_i32",
    "to_dtype_f8e4m3_i64",
    "to_dtype_f8e4m3_u32",
    "to_dtype_f8e4m3_u8",
    "to_dtype_f16_f8e4m3",
    "to_dtype_f32_f8e4m3",
    "to_dtype_f64_f8e4m3",
    "to_dtype_i16_f8e4m3",
    "to_dtype_i32_f8e4m3",
    "to_dtype_i64_f8e4m3",
    "to_dtype_u32_f8e4m3",
    "to_dtype_u8_f8e4m3",
    "to_dtype_bf16_f8e4m3",
    "to_dtype_f8e4m3_bf16",
];

/// Shaders that use the SPIR-V cooperative-matrix load/store ops
/// (`linalg.CoopMat`): compiled with `-capability spirv_1_4` so the
/// buffer pointers land in the `StorageBuffer` storage class. Below
/// SPIR-V 1.4, Slang emits them in the legacy `Uniform` class, which
/// `OpCooperativeMatrixLoadKHR`/`StoreKHR` reject (the spec restricts
/// those ops to the Workgroup, StorageBuffer and PhysicalStorageBuffer
/// classes) and drivers mislower.
const COOPMAT_SHADERS: &[&str] = &[
    "gemm_coopmat_f16",
    "gemm_coopmat_bf16",
    "affine_coopmat_f16",
    "affine_coopmat_bf16",
];

/// Shaders whose bit-for-bit behavior matters: compiled with
/// `-fp-mode precise` so slang does not contract `mul + add` pairs into
/// FMA instructions. The default FP mode permits FMA contraction on the
/// SPIR-V target, which changes the result of the fused norm kernels
/// (e.g. `rms_norm` vs the `rms_norm_slow` reference: the fused kernel
/// sums `v * v` in one loop, while the reference squares with the unary
/// kernel and adds with the reduce kernel, each a separately rounded
/// operation).
const PRECISE_FP_SHADERS: &[&str] = &["rms_norm"];

fn precise_fp_flags(name: &str) -> &'static [&'static str] {
    if PRECISE_FP_SHADERS.iter().any(|s| *s == name) {
        &["-fp-mode", "precise"]
    } else {
        &[]
    }
}

fn coopmat_flags(name: &str) -> &'static [&'static str] {
    if COOPMAT_SHADERS.iter().any(|s| *s == name) {
        &["-capability", "spirv_1_4"]
    } else {
        &[]
    }
}

/// slangc optimization flag for this build, or `None` for the slangc
/// default.
///
/// - `CANDLE_VULKAN_SLANG_OPT`: explicit override. A non-empty value is
///   passed to slangc as-is (e.g. `-O2`, `-O1`); `none`
///   (case-insensitive) or an empty value disables the flag. Takes
///   precedence over everything else.
/// - `CANDLE_VULKAN_SLANG_NO_OPT`: set to any value other than `0` or
///   empty to disable the flag, even when the default below would add
///   one.
/// - Otherwise, when cargo builds the release profile (the `PROFILE`
///   environment variable is `release`), use `-O3`.
/// - Otherwise: no flag (slangc default).
fn slang_opt_flag() -> Option<String> {
    if let Ok(value) = std::env::var("CANDLE_VULKAN_SLANG_OPT") {
        let value = value.trim();
        return if value.is_empty() || value.eq_ignore_ascii_case("none") {
            None
        } else {
            Some(value.to_string())
        };
    }
    if let Ok(value) = std::env::var("CANDLE_VULKAN_SLANG_NO_OPT") {
        let value = value.trim();
        if !value.is_empty() && value != "0" {
            return None;
        }
    }
    if std::env::var("PROFILE").map(|p| p == "release").unwrap_or(false) {
        return Some("-O3".to_string());
    }
    None
}

fn compile_slang(
    source_path: &Path,
    out_path: &Path,
    include_dir: Option<&Path>,
    defines: &[&str],
    extra_flags: &[&str],
) {
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
    // Extra flags (e.g. `-capability spirv_1_4` for the cooperative-matrix
    // shaders).
    for flag in extra_flags {
        args.push(flag.to_string());
    }
    // Include search path so `import` resolves module files that live in
    // `shaders/`; needed for the test shaders under `shaders/test/`.
    if let Some(dir) = include_dir {
        args.push("-I".to_string());
        args.push(dir.to_string_lossy().into_owned());
    }
    // Preprocessor macros (e.g. `BF16_NATIVE=1` for the native bf16 pass).
    for define in defines {
        args.push(format!("-D{define}"));
    }
    // slangc optimization flag (see `slang_opt_flag`): `-O3` for cargo
    // release builds unless overridden or disabled.
    if let Some(flag) = slang_opt_flag() {
        args.push(flag);
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
