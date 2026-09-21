mod common;
use common::all_ops;

use wax_frontend::{KernelCompiler, ast};

/// Manual `#[wax::module]`
fn kernel_ast() -> ast::Module {
    const SRC: &str = r#"
mod gemm_kernels {
    use cutile::core::*;
    #[wax::entry()]
    fn gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = constant(E::ZERO, const_shape![BM, BN]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }
}
"#;
    let item: syn::ItemMod = syn::parse_str(SRC).expect("kernel parses");
    let mut m = ast::Module::with_span_base(
        "gemm_kernels",
        item,
        ast::SpanBase::new("tests/kernel_compilation.rs".into(), 1, 1),
    );
    m.set_absolute_path("wax_frontend::gemm_kernels".to_string());
    m
}

#[test]
fn gemm_kernel_compiles_to_valid_wax_ir() {
    let out = KernelCompiler::new(kernel_ast, "gemm_kernels", "gemm")
        .generics(vec![
            "f32".into(),
            "8".into(),
            "8".into(),
            "8".into(),
            "64".into(),
        ])
        .strides(&[("z", &[8, 1]), ("x", &[64, 1]), ("y", &[8, 1])])
        .compile()
        .expect("kernel compiled");

    assert_eq!(out.module.functions.len(), 1, "single entry point");
    use wax_ir::source::WaxSource;
    let m = out.to_wax_module("gemm").expect("module builds");
    let ops = all_ops(&m);
    assert!(!ops.is_empty(), "module has no ops");

    // Expected ops in the gemm kernel
    let seen: Vec<wax_ir::Opcode> = ops.iter().map(|o| m.opcode(*o)).collect();
    for want in [
        wax_ir::Opcode::MakeTensorView,
        wax_ir::Opcode::MakePartitionView,
        wax_ir::Opcode::For,
        wax_ir::Opcode::LoadViewTko,
        wax_ir::Opcode::MmaF,
        wax_ir::Opcode::StoreViewTko,
    ] {
        assert!(seen.contains(&want), "{want:?} missing from IR");
    }
}

#[test]
fn unknown_entry_point_is_an_error() {
    let err = match KernelCompiler::new(kernel_ast, "gemm_kernels", "wrong_kernel_name")
        .generics(vec![
            "f32".into(),
            "8".into(),
            "8".into(),
            "8".into(),
            "64".into(),
        ])
        .compile()
    {
        Ok(_) => panic!("Should be unreachable. No kernel names 'wrong_kernel_name'."),
        Err(e) => e,
    };
    assert!(
        err.contains("wrong_kernel_name"),
        "error should contain missing kernel name. Got: {err}"
    );
}
