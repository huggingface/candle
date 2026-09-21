//! What the conversion cannot yet handle, on a straight-line kernel.

use wax_frontend::{ast, KernelCompiler};

const SCALE_SRC: &str = r#"
mod k {
    use cutile::core::*;
    #[wax::entry()]
    fn scale<const N: i32>(out: &mut Tensor<f32, { [1, N] }>, x: &Tensor<f32, { [1, N] }>) {
        let ts = const_shape![1, N];
        let px = x.partition(ts);
        let mut po: PartitionMut<f32, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
        let v = px.load([0i32, 0i32]);
        unsafe { po.store(v + v, [0i32, 0i32]) };
    }
}
"#;

fn module() -> wax_ir::Module {
    let item: syn::ItemMod = syn::parse_str(SCALE_SRC).expect("kernel parses");
    let span_base = ast::SpanBase::new("tests/wax_to_air_worklist.rs".into(), 1, 1);
    KernelCompiler::new(
        move || {
            let mut m = ast::Module::with_span_base("k", item.clone(), span_base.clone());
            m.set_absolute_path("wax_metal::k".to_string());
            m
        },
        "k",
        "scale",
    )
    .generics(vec!["32".into()])
    .strides(&[("out", &[32, 1]), ("x", &[32, 1])])
    .compile()
    .unwrap()
    .to_wax_module("scale")
    .unwrap()
}

/// A straight-line kernel must convert with nothing left over.
///
/// The work list is the point: an op with no rewrite is recorded rather than silently left in
/// place, so this test names exactly what is missing when it fails instead of failing somewhere
/// downstream in the emitter.
#[test]
fn a_straight_line_kernel_converts_whole() {
    let mut m = module();
    let f = *m.functions.first().unwrap();
    let (c, left) =
        wax_metal::wax_to_air::convert_reporting(&mut m.ctx, f).unwrap();
    assert!(c.converted > 0, "nothing converted");
    assert!(
        left.is_empty(),
        "{} ops converted, but these have no rewrite: {left}",
        c.converted
    );
}

/// The converted function must carry the parameters the AIR function has.
///
/// `%_tp` and `%_tg` are appended by `lower/walk.rs` when it builds the CFG; the wax function
/// has neither. Metal binds by position, so they must come last and in that order.
#[test]
fn the_converted_function_gains_the_implicit_parameters() {
    let mut m = module();
    let f = *m.functions.first().unwrap();

    wax_metal::wax_to_air::convert_reporting(&mut m.ctx, f).unwrap();

    // Counting is the wrong assertion: the signature rewrite both appends `%_tp`/`%_tg` and
    // DROPS the scalar-tile params the AIR signature does not carry. What matters is that the
    // two position vectors are present and last, because Metal binds by position.
    let tys = entry_arg_types(&m, f);
    assert!(tys.len() >= 2, "signature is too short: {tys:?}");
    assert_eq!(
        &tys[tys.len() - 2..],
        &["<3 x i32>".to_string(), "<3 x i32>".to_string()],
        "expected `%_tp` and `%_tg` last, signature is {tys:?}"
    );
    assert!(
        !tys[..tys.len() - 2].iter().any(|t| t.starts_with("<3 x")),
        "a position vector appears before the trailing pair: {tys:?}"
    );
}

/// The entry block's argument types, rendered.
fn entry_arg_types(m: &wax_ir::Module, f: wax_ir::build::OpId) -> Vec<String> {
    use pliron::linked_list::ContainsLinkedList;
    use pliron::r#type::Typed;
    let r = f
        .deref(&m.ctx)
        .regions()
        .next()
        .expect("function has a region");
    let b = r
        .deref(&m.ctx)
        .iter(&m.ctx)
        .next()
        .expect("region has an entry block");
    let b = b.deref(&m.ctx);
    (0..b.get_num_arguments())
        .map(|i| {
            wax_metal::air_dialect::types::from_pliron(
                &m.ctx,
                b.get_argument(i).get_type(&m.ctx),
            )
            .map(|ll| ll.ll_str())
            .unwrap_or_else(|| "<unconverted>".into())
        })
        .collect()
}