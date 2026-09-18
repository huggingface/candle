//! Tests for AtomicRMW and AtomicCAS lowering.
//!
//! Builds minimal wax_ir::Module instances directly (no DSL compiler pass) and
//! verifies that lower_to_air() produces the expected AIR intrinsic calls, then
//! assembles the output with xcrun metal-as to confirm it is valid AIR.

use std::process::Command;

use wax_ir::Opcode;
use wax_ir::builder::{OpBuilder, append_op};
use wax_ir::{
    Attribute, Location, Module, PointerType, ScalarType, TileElementType, TileType, Type,
};
use wax_metal::lower_to_air;

// helpers

fn xcrun_metal_as(ll: &str, tag: &str) -> bool {
    let tmp = std::env::temp_dir();
    let ll_path = tmp.join(format!("{tag}.ll"));
    let air_path = tmp.join(format!("{tag}.air"));
    std::fs::write(&ll_path, ll).unwrap();
    let out = Command::new("xcrun")
        .args([
            "metal-as",
            ll_path.to_str().unwrap(),
            "-o",
            air_path.to_str().unwrap(),
        ])
        .output()
        .expect("xcrun metal-as");
    if !out.status.success() {
        eprintln!("metal-as stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    }
    out.status.success()
}

/// Build a minimal module: one kernel `atomic_add_test(ptr: *mut i32, val: i32)`.
///
/// Body:
///   old = AtomicRMW(mode=Add, ordering=Relaxed, scope=Device) ptr, val
///   return
fn build_atomic_rmw_module() -> Module {
    let mut m = Module::new("atomic_rmw_test");

    let ptr_ty = Type::Pointer(PointerType {
        pointee: ScalarType::I32,
    });
    let scalar_i32 = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::I32),
    });

    let (entry_id, block_id, args) =
        m.begin_function("atomic_add_test", &[ptr_ty, scalar_i32.clone()]);
    let ptr_arg = args[0];
    let val_arg = args[1];

    // AtomicRMW: mode=3 (Add), ordering=Relaxed (1), scope=Device (1)
    let (rmw_id, _rmw_results) = OpBuilder::new(Opcode::AtomicRMW, Location::Unknown)
        .operands([ptr_arg, val_arg])
        .result(scalar_i32.clone()) // result[0]: old value
        .result(Type::Token) // result[1]: token
        .attr("mode", Attribute::i32(3)) // Add
        .attr("memory_ordering_semantics", Attribute::i32(1)) // Relaxed
        .attr("memory_scope", Attribute::i32(1)) // Device
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(0),
                Attribute::i32(0),
            ]),
        )
        .build(&mut m);
    append_op(&mut m, block_id, rmw_id);

    let (ret_id, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut m);
    append_op(&mut m, block_id, ret_id);

    m.set_op_attr(
        entry_id,
        "sym_name",
        Attribute::String("atomic_add_test".into()),
    );
    m.functions.push(entry_id);

    m
}

/// Build a minimal module: one kernel `atomic_cas_test(ptr: *mut i32, cmp: i32, new: i32)`.
fn build_atomic_cas_module() -> Module {
    let mut m = Module::new("atomic_cas_test");

    let ptr_ty = Type::Pointer(PointerType {
        pointee: ScalarType::I32,
    });
    let scalar_i32 = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::I32),
    });

    let (entry_id, block_id, args) = m.begin_function(
        "atomic_cas_test",
        &[ptr_ty, scalar_i32.clone(), scalar_i32.clone()],
    );
    let ptr_arg = args[0];
    let cmp_arg = args[1];
    let new_arg = args[2];

    // AtomicCAS: ordering=Relaxed (1), scope=Device (1)
    let (cas_id, _cas_results) = OpBuilder::new(Opcode::AtomicCAS, Location::Unknown)
        .operands([ptr_arg, cmp_arg, new_arg])
        .result(scalar_i32.clone()) // result[0]: old value
        .result(Type::Token) // result[1]: token
        .attr("memory_ordering_semantics", Attribute::i32(1))
        .attr("memory_scope", Attribute::i32(1))
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(0),
                Attribute::i32(0),
            ]),
        )
        .build(&mut m);
    append_op(&mut m, block_id, cas_id);

    let (ret_id, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut m);
    append_op(&mut m, block_id, ret_id);

    m.set_op_attr(
        entry_id,
        "sym_name",
        Attribute::String("atomic_cas_test".into()),
    );
    m.functions.push(entry_id);

    m
}

// tests

#[test]
fn atomic_rmw_lowers_to_air_add() {
    let m = build_atomic_rmw_module();
    let ir = lower_to_air(&m).expect("lower_to_air");
    println!("--- AtomicRMW AIR ---\n{ir}");
    assert!(
        ir.contains("air.atomic.global.add.s.i32"),
        "expected @air.atomic.global.add.s.i32 in emitted IR\n{ir}"
    );
    assert!(
        xcrun_metal_as(&ir, "atomic_rmw_add"),
        "AIR from AtomicRMW did not assemble"
    );
}

#[test]
fn atomic_cas_lowers_to_air_cmpxchg() {
    let m = build_atomic_cas_module();
    let ir = lower_to_air(&m).expect("lower_to_air");
    println!("--- AtomicCAS AIR ---\n{ir}");
    assert!(
        ir.contains("air.atomic.global.cmpxchg.weak.i32"),
        "expected @air.atomic.global.cmpxchg.weak.i32 in emitted IR\n{ir}"
    );
    assert!(
        xcrun_metal_as(&ir, "atomic_cas_cmpxchg"),
        "AIR from AtomicCAS did not assemble"
    );
}
