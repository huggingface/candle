//! Validate that f16 and bf16 AIR patterns produced by the lowerer are accepted by xcrun metal-as.
//!
//! These tests hand-author the specific AIR snippets the lowerer would emit for half-precision
//! kernels and confirm they assemble correctly.  They cover:
//!   - f16 load / store
//!   - @llvm.exp.f16, @llvm.sqrt.f16
//!   - fpext/simd_sum.f32/fptrunc reduction pattern
//!   - fcmp ogt half / select for MaxF
//!   - half 0xH... constant literals
//!   - bfloat 0xR... constant literals

use std::process::Command;

use wax_metal::TargetProfile;

fn metal_as_ok(ir: &str, tag: &str) -> bool {
    let tmp = std::env::temp_dir();
    let ll = tmp.join(format!("{tag}.ll"));
    let air = tmp.join(format!("{tag}.air"));
    std::fs::write(&ll, ir).unwrap();
    let ok = Command::new("xcrun")
        .args([
            "metal-as",
            ll.to_str().unwrap(),
            "-o",
            air.to_str().unwrap(),
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        // Print stderr for diagnosis.
        let _ = Command::new("xcrun")
            .args([
                "metal-as",
                ll.to_str().unwrap(),
                "-o",
                air.to_str().unwrap(),
            ])
            .output()
            .map(|o| eprintln!("{}", String::from_utf8_lossy(&o.stderr)));
    }
    ok
}

// Shared header

/// Datalayout + triple for the target this build emits, so these hand-written modules
/// are assembled for the same target the lowerer would have picked.
fn air_header() -> String {
    let p = TargetProfile::host();
    format!(
        "target datalayout = \"{}\"\ntarget triple = \"{}\"\n\n",
        p.datalayout(),
        p.triple()
    )
}

/// The three version metadata nodes, numbered `!{sdk}`, `!{air}` and `!{lang}`.
/// Pinning these while the triple follows the profile would make the module
/// self-contradictory, and metal-as checks.
fn version_nodes(sdk: u32, air: u32, lang: u32) -> String {
    let p = TargetProfile::host();
    let (sdk_major, sdk_minor) = p.sdk_version();
    let (a0, a1, a2) = p.air_version;
    let (l0, l1, l2) = p.language_version;
    format!(
        "!{sdk} = !{{i32 2, !\"SDK Version\", [2 x i32] [i32 {sdk_major}, i32 {sdk_minor}]}}\n\
         !{air} = !{{i32 {a0}, i32 {a1}, i32 {a2}}}\n\
         !{lang} = !{{!\"Metal\", i32 {l0}, i32 {l1}, i32 {l2}}}\n"
    )
}

fn attrs() -> String {
    version_nodes(1, 3, 4)
        + concat!(
            "\nattributes #0 = { nounwind }\n",
            "attributes #1 = { nounwind readnone }\n\n",
            "!air.kernel = !{!0}\n",
            "!llvm.module.flags = !{!1, !2}\n",
            "!air.version = !{!3}\n",
            "!air.language_version = !{!4}\n",
            "!llvm.ident = !{!5}\n",
            "!0 = !{void (half addrspace(1)*, half addrspace(1)*, <3 x i32>, <3 x i32>)* @f16_kernel, !6, !7}\n",
            "!2 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n",
            "!5 = !{!\"wax-metal\"}\n",
            "!6 = !{}\n",
            "!7 = !{!8, !9, !10, !11}\n",
            "!8  = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\",\
     !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 2, !\"air.arg_type_align_size\", i32 2,\
     !\"air.arg_type_name\", !\"half\", !\"air.arg_name\", !\"param0\"}\n",
            "!9  = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\",\
     !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 2, !\"air.arg_type_align_size\", i32 2,\
     !\"air.arg_type_name\", !\"half\", !\"air.arg_name\", !\"param1\"}\n",
            "!10 = !{i32 2, !\"air.thread_position_in_threadgroup\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"_tp\"}\n",
            "!11 = !{i32 3, !\"air.threadgroup_position_in_grid\",   !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"_tg\"}\n",
        )
}

// f16 exp + load/store

#[test]
fn f16_exp_load_store_compiles() {
    let header = air_header();
    let attrs = attrs();
    let ir = format!(
        r#"{header}
declare half @llvm.exp.f16(half) #1
declare half @llvm.sqrt.f16(half) #1
declare float @air.simd_sum.f32(float) #1

define void @f16_kernel(
    half addrspace(1)* %p0,
    half addrspace(1)* %p1,
    <3 x i32> %_tp,
    <3 x i32> %_tg) #0 {{
entry:
    %lane = extractelement <3 x i32> %_tp, i32 0
    %bid  = extractelement <3 x i32> %_tg, i32 0
    ; flat index: bid * 32 + lane
    %base = mul i32 %bid, 32
    %flat = add i32 %base, %lane
    ; load f16
    %ptr_in  = getelementptr half, half addrspace(1)* %p0, i32 %flat
    %x       = load half, half addrspace(1)* %ptr_in, align 2
    ; exp(x) - @llvm.exp.f16
    %ex = call half @llvm.exp.f16(half %x)
    ; store
    %ptr_out = getelementptr half, half addrspace(1)* %p1, i32 %flat
    store half %ex, half addrspace(1)* %ptr_out, align 2
    ret void
}}
{attrs}"#
    );
    assert!(
        metal_as_ok(&ir, "dtype_f16_exp"),
        "metal-as rejected f16 exp kernel"
    );
}

// f16 reduction via fpext/simd_sum.f32/fptrunc

#[test]
fn f16_reduction_upcast_compiles() {
    let header = air_header();
    let attrs = attrs();
    let ir = format!(
        r#"{header}
declare float @air.simd_sum.f32(float) #1
declare float @air.simd_max.f32(float) #1

define void @f16_kernel(
    half addrspace(1)* %p0,
    half addrspace(1)* %p1,
    <3 x i32> %_tp,
    <3 x i32> %_tg) #0 {{
entry:
    %lane = extractelement <3 x i32> %_tp, i32 0
    %ptr_in = getelementptr half, half addrspace(1)* %p0, i32 %lane
    %x = load half, half addrspace(1)* %ptr_in, align 2
    ; reduction via upcast - the pattern emitted by Opcode::Reduce for non-f32
    %xf  = fpext half %x to float
    %sum = call fast float @air.simd_sum.f32(float %xf)
    %r   = fptrunc float %sum to half
    ; store result
    %ptr_out = getelementptr half, half addrspace(1)* %p1, i32 %lane
    store half %r, half addrspace(1)* %ptr_out, align 2
    ret void
}}
{attrs}"#
    );
    assert!(
        metal_as_ok(&ir, "dtype_f16_reduction"),
        "metal-as rejected f16 reduction upcast pattern"
    );
}

// f16 fcmp / MaxF pattern

#[test]
fn f16_maxf_compiles() {
    let header = air_header();
    let attrs = attrs();
    let ir = format!(
        r#"{header}

define void @f16_kernel(
    half addrspace(1)* %p0,
    half addrspace(1)* %p1,
    <3 x i32> %_tp,
    <3 x i32> %_tg) #0 {{
entry:
    %lane = extractelement <3 x i32> %_tp, i32 0
    %i0 = mul i32 %lane, 2
    %i1 = add i32 %i0, 1
    %pa = getelementptr half, half addrspace(1)* %p0, i32 %i0
    %pb = getelementptr half, half addrspace(1)* %p0, i32 %i1
    %a  = load half, half addrspace(1)* %pa, align 2
    %b  = load half, half addrspace(1)* %pb, align 2
    ; MaxF pattern - fcmp ogt half + select
    %cmp = fcmp ogt half %a, %b
    %r   = select i1 %cmp, half %a, half %b
    %ptr_out = getelementptr half, half addrspace(1)* %p1, i32 %lane
    store half %r, half addrspace(1)* %ptr_out, align 2
    ret void
}}
{attrs}"#
    );
    assert!(
        metal_as_ok(&ir, "dtype_f16_maxf"),
        "metal-as rejected f16 MaxF (fcmp ogt half) pattern"
    );
}

// half 0xH... and bfloat 0xR... constant literals

#[test]
fn f16_constant_literal_compiles() {
    // Exercises the constant literal paths: half 0xH3C00 (1.0), half 0xH0000 (0.0)
    let header = air_header();
    let attrs = attrs();
    let ir = format!(
        r#"{header}

define void @f16_kernel(
    half addrspace(1)* %p0,
    half addrspace(1)* %p1,
    <3 x i32> %_tp,
    <3 x i32> %_tg) #0 {{
entry:
    %lane = extractelement <3 x i32> %_tp, i32 0
    %ptr_in = getelementptr half, half addrspace(1)* %p0, i32 %lane
    %x = load half, half addrspace(1)* %ptr_in, align 2
    ; Use half constants - 0xH3C00 = 1.0f16, 0xH0000 = 0.0f16
    %y = fadd half %x, 0xH3C00
    %z = fadd half %y, 0xH0000
    %ptr_out = getelementptr half, half addrspace(1)* %p1, i32 %lane
    store half %z, half addrspace(1)* %ptr_out, align 2
    ret void
}}
{attrs}"#
    );
    assert!(
        metal_as_ok(&ir, "dtype_f16_const"),
        "metal-as rejected half 0xH... constant literal"
    );
}

// bfloat16 basic patterns

#[test]
fn bf16_load_store_compiles() {
    // Reuse the same metadata with bfloat instead of half.
    // Build a bfloat-specific metadata block (arg_type_size = 2, arg_type_name = "bfloat").
    let ir = air_header()
        + concat!(
            "declare bfloat @llvm.exp.bf16(bfloat) #1\n\n",
            "define void @bf16_kernel(\n",
            "    bfloat addrspace(1)* %p0,\n",
            "    bfloat addrspace(1)* %p1,\n",
            "    <3 x i32> %_tp,\n",
            "    <3 x i32> %_tg) #0 {\n",
            "entry:\n",
            "    %lane = extractelement <3 x i32> %_tp, i32 0\n",
            "    %ptr_in  = getelementptr bfloat, bfloat addrspace(1)* %p0, i32 %lane\n",
            "    %x       = load bfloat, bfloat addrspace(1)* %ptr_in, align 2\n",
            "    ; bfloat constant 0xR3F80 = 1.0 in bfloat16\n",
            "    %y = fadd bfloat %x, 0xR3F80\n",
            "    %ptr_out = getelementptr bfloat, bfloat addrspace(1)* %p1, i32 %lane\n",
            "    store bfloat %y, bfloat addrspace(1)* %ptr_out, align 2\n",
            "    ret void\n",
            "}\n\n",
            "attributes #0 = { nounwind }\n",
            "attributes #1 = { nounwind readnone }\n\n",
            "!air.kernel = !{!0}\n",
            "!llvm.module.flags = !{!1, !2}\n",
            "!air.version = !{!3}\n",
            "!air.language_version = !{!4}\n",
            "!llvm.ident = !{!5}\n",
            "!0 = !{void (bfloat addrspace(1)*, bfloat addrspace(1)*, <3 x i32>, <3 x i32>)* @bf16_kernel, !6, !7}\n",
            "!2 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n",
            "!5 = !{!\"wax-metal\"}\n",
            "!6 = !{}\n",
            "!7 = !{!8, !9, !10, !11}\n",
            "!8  = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\",",
            " !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 2, !\"air.arg_type_align_size\", i32 2,",
            " !\"air.arg_type_name\", !\"bfloat\", !\"air.arg_name\", !\"param0\"}\n",
            "!9  = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\",",
            " !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 2, !\"air.arg_type_align_size\", i32 2,",
            " !\"air.arg_type_name\", !\"bfloat\", !\"air.arg_name\", !\"param1\"}\n",
            "!10 = !{i32 2, !\"air.thread_position_in_threadgroup\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"_tp\"}\n",
            "!11 = !{i32 3, !\"air.threadgroup_position_in_grid\",   !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"_tg\"}\n",
        )
        + &version_nodes(1, 3, 4);
    assert!(
        metal_as_ok(&ir, "dtype_bf16"),
        "metal-as rejected bfloat16 load/store/constant pattern"
    );
}

// SGMMA f16 load / MMA / store

#[test]
fn sgmma_f16_load_mma_store_compiles() {
    // Validates the Phase-2 AIR patterns for f16 SGMMA:
    //   @air.simdgroup_matrix_8x8_load.v64f32.p1f16   - load half* -> <64 x float>
    //   @air.simdgroup_matrix_8x8_store.v64f32.p1f32  - store <64 x float> -> float* (f32 output)
    //   @air.simdgroup_matrix_8x8_store.v64f32.p1f16  - store <64 x float> -> half* (f16 output)
    // The MMA accumulate intrinsic is dtype-independent (always v64f32).
    let ir = air_header()
        + concat!(
            "declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f16(\
         half addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)\n",
            "declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(\
         float addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)\n",
            "declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate\
         .v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)\n",
            "declare void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(\
         <64 x float>, float addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)\n",
            "declare void @air.simdgroup_matrix_8x8_store.v64f32.p1f16(\
         <64 x float>, half addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)\n",
            "declare <64 x float> @air.simdgroup_matrix_8x8_init_diag.v64f32.f32(float)\n\n",
            // Kernel: load two f16 tiles, MMA, store f32 result and f16 result.
            "define void @sgmma_f16_kernel(\n",
            "    half addrspace(1)* %xa,\n",
            "    half addrspace(1)* %xb,\n",
            "    float addrspace(1)* %zf,\n",
            "    half addrspace(1)* %zh,\n",
            "    <3 x i32> %_tp, <3 x i32> %_tg) #0 {\n",
            "entry:\n",
            "  %acc = call <64 x float> @air.simdgroup_matrix_8x8_init_diag.v64f32.f32(float 0.0)\n",
            "  %a = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f16(\n",
            "      half addrspace(1)* %xa,\n",
            "      <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)\n",
            "  %b = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f16(\n",
            "      half addrspace(1)* %xb,\n",
            "      <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)\n",
            "  %c = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate\
         .v64f32.v64f32.v64f32.v64f32(<64 x float> %a, <64 x float> %b, <64 x float> %acc)\n",
            "  call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(\n",
            "      <64 x float> %c, float addrspace(1)* %zf,\n",
            "      <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)\n",
            "  call void @air.simdgroup_matrix_8x8_store.v64f32.p1f16(\n",
            "      <64 x float> %c, half addrspace(1)* %zh,\n",
            "      <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)\n",
            "  ret void\n",
            "}\n\n",
            "attributes #0 = { nounwind }\n\n",
            "!air.kernel = !{!0}\n",
            "!llvm.module.flags = !{!1, !2}\n",
            "!air.version = !{!3}\n",
            "!air.language_version = !{!4}\n",
            "!llvm.ident = !{!5}\n",
            "!0 = !{void (half addrspace(1)*, half addrspace(1)*, float addrspace(1)*, half addrspace(1)*,\
         <3 x i32>, <3 x i32>)* @sgmma_f16_kernel, !6, !7}\n",
            "!2 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n",
            "!5 = !{!\"wax-metal\"}\n",
            "!6 = !{}\n",
            "!7 = !{!8, !9, !10, !11, !12, !13}\n",
            "!8  = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\",\
         !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 2, !\"air.arg_type_align_size\", i32 2,\
         !\"air.arg_type_name\", !\"half\", !\"air.arg_name\", !\"xa\"}\n",
            "!9  = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\",\
         !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 2, !\"air.arg_type_align_size\", i32 2,\
         !\"air.arg_type_name\", !\"half\", !\"air.arg_name\", !\"xb\"}\n",
            "!10 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read_write\",\
         !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4,\
         !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"zf\"}\n",
            "!11 = !{i32 3, !\"air.buffer\", !\"air.location_index\", i32 3, i32 1, !\"air.read_write\",\
         !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 2, !\"air.arg_type_align_size\", i32 2,\
         !\"air.arg_type_name\", !\"half\", !\"air.arg_name\", !\"zh\"}\n",
            "!12 = !{i32 4, !\"air.thread_position_in_threadgroup\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"_tp\"}\n",
            "!13 = !{i32 5, !\"air.threadgroup_position_in_grid\",   !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"_tg\"}\n",
        )
        + &version_nodes(1, 3, 4);
    assert!(
        metal_as_ok(&ir, "dtype_sgmma_f16"),
        "metal-as rejected f16 SGMMA load/MMA/store pattern"
    );
}
