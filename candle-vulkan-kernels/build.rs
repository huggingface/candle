use std::fs;
use std::path::Path;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let src_dir = Path::new("shaders");
    for shader in ["affine", "reduce"] {
        let source = fs::read_to_string(src_dir.join(format!("{shader}.comp"))).unwrap_or_else(|e| {
            panic!("failed to read shaders/{shader}.comp: {e}")
        });
        let module = parse_glsl(&source);
        let words = to_spv(&module);
        let out_path = Path::new(&out_dir).join(format!("{shader}.spv"));
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
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

fn to_spv(module: &naga::Module) -> Vec<u32> {
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(module)
        .expect("naga validation failed");
    let spv_options = naga::back::spv::Options {
        lang_version: (1, 2),
        ..Default::default()
    };
    naga::back::spv::write_vec(
        module,
        &info,
        &spv_options,
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".to_string(),
        }),
    )
    .expect("failed to write SPIR-V")
}
