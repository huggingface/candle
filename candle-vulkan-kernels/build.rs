use std::fs;
use std::path::Path;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let src_dir = Path::new("shaders");
    // `elementwise` and `gemm` are WGSL (they carry one entry point per op,
    // which naga's GLSL frontend cannot express); `affine` and `reduce` are
    // GLSL. All sources live under the `.comp` extension.
    for (shader, lang) in [
        ("affine", "glsl"),
        ("reduce", "glsl"),
        ("reduce_max", "glsl"),
        ("gather", "glsl"),
        ("copy", "glsl"),
        ("rms_norm", "wgsl"),
        ("softmax", "wgsl"),
        ("rope", "wgsl"),
        ("gemv", "wgsl"),
        ("gemv_t", "wgsl"),
        ("elementwise", "wgsl"),
        ("gemm", "wgsl"),
        ("q4k", "wgsl"),
    ] {
        let source_path = src_dir.join(format!("{shader}.comp"));
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|e| panic!("failed to read shaders/{shader}.comp: {e}"));
        let module = if lang == "wgsl" {
            parse_wgsl(&source)
        } else {
            parse_glsl(&source)
        };
        let words = to_spv(&module);
        let out_path = Path::new(&out_dir).join(format!("{shader}.spv"));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        fs::write(&out_path, &bytes).expect("failed to write SPIR-V");
        println!("cargo:rerun-if-changed=shaders/{shader}.comp");
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
