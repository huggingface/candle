use std::fs;
use std::path::Path;

#[derive(PartialEq)]
enum ShaderLang {
    Glsl,
    Wgsl,
    Slang,
}

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let src_dir = Path::new("shaders");
    // `elementwise` and `gemm` are WGSL (they carry one entry point per op,
    // which naga's GLSL frontend cannot express); `affine` and `reduce` are
    // GLSL. All sources live under the `.comp` extension.
    for (shader, lang) in [
        ("affine", ShaderLang::Glsl),
        ("reduce", ShaderLang::Glsl),
        ("reduce_max", ShaderLang::Glsl),
        ("gather", ShaderLang::Glsl),
        ("copy", ShaderLang::Glsl),
        ("rms_norm", ShaderLang::Wgsl),
        ("softmax", ShaderLang::Wgsl),
        ("rope", ShaderLang::Wgsl),
        ("gemv", ShaderLang::Wgsl),
        ("gemv_t", ShaderLang::Wgsl),
        ("elementwise", ShaderLang::Wgsl),
        ("gemm", ShaderLang::Wgsl),
        ("q4k", ShaderLang::Wgsl),
        ("q5q8", ShaderLang::Wgsl),
        ("test_fill_f16", ShaderLang::Slang),
    ] {
        let source_path = match lang {
            ShaderLang::Slang => src_dir.join(format!("{shader}.slang")),
            _ => src_dir.join(format!("{shader}.comp")),
        };
        let out_path = Path::new(&out_dir).join(format!("{shader}.spv"));
        match lang {
            ShaderLang::Glsl | ShaderLang::Wgsl => {
                let source = fs::read_to_string(&source_path)
                    .unwrap_or_else(|e| panic!("failed to read shaders/{shader}.comp: {e}"));
                let module = if ShaderLang::Glsl.eq(&lang) {
                    parse_glsl(&source)
                } else {
                    parse_wgsl(&source)
                };
                let spirv_ir = to_spv(&module);
                let bytes: Vec<u8> = spirv_ir.iter().flat_map(|w| w.to_le_bytes()).collect();
                fs::write(&out_path, &bytes).expect("failed to write SPIR-V");
            },
            ShaderLang::Slang => {
                let exe = if cfg!(windows) {
                    ".exe"
                } else {
                    ""
                };
                let slangc = format!("slangc{exe}");
                let status = std::process::Command::new(&slangc)
                    .args(&[
                        source_path.to_string_lossy().into_owned().as_str(),
                        "-profile",
                        "glsl_450",
                        "-target",
                        "spirv",
                        "-o",
                        out_path.to_string_lossy().into_owned().as_str(),
                        "-entry",
                        "main",
                    ])
                    .stdout(std::process::Stdio::piped())
                    .status()
                    .unwrap();

                if !status.success() {
                    panic!("slangc failed compiling source '{source_path:?}'.");
                }
            },
        };
        let ext = match lang {
            ShaderLang::Slang => "slang",
            _ => "comp",
        };
        println!("cargo:rerun-if-changed=shaders/{shader}.{ext}");
    }
}

fn parse_glsl(source: &str) -> naga::Module {
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options::from(naga::ShaderStage::Compute);
    frontend
        .parse(&options, source)
        .expect("failed to parse GLSL with naga")
}

fn parse_wgsl(source: &str) -> naga::Module {
    naga::front::wgsl::parse_str(source).expect("failed to parse WGSL with naga")
}

fn to_spv(module: &naga::Module) -> Vec<u32> {
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(module).expect("naga validation failed");
    let spv_options = naga::back::spv::Options {
        lang_version: (1, 2),
        ..Default::default()
    };
    // No entry-point filter: `elementwise` holds one entry point per op and
    // the pipeline cache selects them by name at runtime; single-entry
    // modules emit their sole entry point.
    naga::back::spv::write_vec(module, &info, &spv_options, None).expect("failed to write SPIR-V")
}
