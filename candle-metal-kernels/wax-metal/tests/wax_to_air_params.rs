//! The conversion's PARAMETER LIST against the CFG path's, for a padded store.
//!
//! A padded store needs the tensor's runtime extent, which the CFG path promotes to an
//! `i32 addrspace(2)*` parameter it calls `%_dim_arg_i`. The signature is checked separately
//! from the bytes because it is an ABI: a missing or extra parameter is a wrong answer at
//! dispatch rather than a compile error, and it is decided by the promotion alone.


use wax_frontend::{ast, KernelCompiler};

const PADDED_STORE_SRC: &str = r#"
mod kernels {
    use cutile::core::*;
    /// A store through a `padding::Zero` view of a runtime-shaped tensor.
    #[wax::entry()]
    fn padded_store<const M: i32>(x: &Tensor<f32, { [-1] }>, out: &mut Tensor<f32, { [M] }>) {
        let ts = const_shape![32];
        let blk = get_tile_block_id().0;
        let t: Tile<f32, { [32] }> = load_tile(x, ts, [blk]);
        let tok: Token = get_tensor_token(out);
        let mut op: PartitionMut<f32, { [32] }> =
            unsafe { make_partition_view_mut(out, ts, padding::Zero, tok) };
        unsafe { op.store(t, [blk]) };
    }
}
"#;

fn module() -> wax_ir::Module {
    let item: syn::ItemMod = syn::parse_str(PADDED_STORE_SRC).expect("kernel parses");
    let span_base = ast::SpanBase::new("tests/wax_to_air_params.rs".into(), 1, 1);
    let m = KernelCompiler::new(
        move || {
            let mut m = ast::Module::with_span_base("kernels", item.clone(), span_base.clone());
            m.set_absolute_path("wax_metal::kernels".to_string());
            m
        },
        "kernels",
        "padded_store",
    )
    .generics(vec!["64".into()])
    .strides(&[("x", &[1i32]), ("out", &[1i32])])
    .compile()
    .expect("frontend compiles padded_store")
    .to_wax_module("padded_store")
    .expect("dialect module builds");
    m
}

#[test]
fn a_padded_store_promotes_its_runtime_extent() {
    let m = module();

    let params = wax_metal::wax_to_air::converted_param_types(&m)
        .expect("conversion produces a signature");

    // Two buffers, then the promoted extent in constant space, then the two position vectors.
    let vec3 = wax_metal::ir_builder::LlType::vec(3, wax_metal::ir_builder::LlType::I32);
    let want_tail = [
        wax_metal::ir_builder::LlType::const_ptr(wax_metal::ir_builder::LlType::I32),
        vec3.clone(),
        vec3,
    ];
    assert!(
        params.len() >= want_tail.len(),
        "signature is shorter than the implicit tail: {params:?}"
    );
    let tail = &params[params.len() - want_tail.len()..];
    assert_eq!(
        tail,
        &want_tail[..],
        "expected the promoted extent then _tp/_tg; got {params:?}"
    );
    // The buffers come first and no position vector appears among them.
    for (i, p) in params[..params.len() - want_tail.len()].iter().enumerate() {
        assert!(
            matches!(p, wax_metal::ir_builder::LlType::Ptr { .. }),
            "parameter {i} should be a buffer pointer, got {p:?}"
        );
    }
}

/// The whole kernel, as bytes: promotion, bounds predicate and store guard together.
#[test]
fn a_padded_store_is_byte_identical() {
    use wax_metal::profile::TargetProfile;
    let m = module();
    let via_cfg = wax_metal::lower_to_metallib(&m).expect("CFG path lowers padded_store");
    let func = *m.functions.first().expect("module has a function");
    let args = wax_metal::lower::kernel_args_for_module(&m, func);
    let bytes =
        wax_metal::wax_to_air::lower_to_metallib_via_conversion(&m, &args, &[], &TargetProfile::host())
            .expect("conversion route lowers padded_store");
    assert_eq!(
        bytes.len(),
        via_cfg.len(),
        "conversion {} bytes vs CFG {}",
        bytes.len(),
        via_cfg.len()
    );
    assert_eq!(bytes, via_cfg, "same length, different bytes");
}