//! AIR emission for Apple's MPP cooperative-tensor GEMM (Metal 4).
//!
//! A second lowering path alongside SGMMA. `mpp::tensor_ops::matmul2d` is a different
//! matmul primitive and is faster: 15.2-15.3 TF bf16 on an M4 Max against 14.5 for our
//! SGMMA kernels.
//!
//! NOTHING NEEDS LINKING. The matmul entry points are emitted as
//! `section "air.externally_defined"` declarations and the DRIVER resolves them at
//! pipeline creation, so a metallib with `__tensorops_impl_matmul2d_*` still undefined
//! creates a pipeline fine.
//!
//! The op runs the whole K loop internally (`BK` is `dynamic_extent`), so tile choice
//! barely matters - 32x32/1 group through 128x128/8 land within ~1.5%. Accumulation is
//! f32 regardless of destination type, so the bf16/f16 destination form is used: it
//! stores straight to memory with no copy loop and no second cooperative tensor.
//!
//! Two emitters produce the same kernel: [`emit_mpp_gemm`] writes AIR text (needs
//! `xcrun metal-as`), [`emit_mpp_gemm_metallib`] writes a metallib in process so MPP works
//! where no Metal toolchain exists. `tests/mpp_gemm.rs` runs every case through both.

use crate::profile::TargetProfile;

/// Element type of the GEMM operands and destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MppDtype {
    BF16,
    F16,
    F32,
}

impl MppDtype {
    fn ll(self) -> &'static str {
        match self {
            MppDtype::BF16 => "bfloat",
            MppDtype::F16 => "half",
            MppDtype::F32 => "float",
        }
    }

    /// Suffix in the run/store symbol names (`..._run_cooperative_dv_b16_dv_b16_b16`).
    fn suffix(self) -> &'static str {
        match self {
            MppDtype::BF16 => "b16",
            MppDtype::F16 => "f16",
            MppDtype::F32 => "f32",
        }
    }

    /// Element type tag the `__tensorops_impl_*` entry points take. Read off the
    /// frontend's own output: 0x90000010 bfloat, 0x10000010 half, 0x10000020 float.
    fn tag(self) -> i32 {
        match self {
            MppDtype::BF16 => -1879048176,
            MppDtype::F16 => 268435472,
            MppDtype::F32 => 268435488,
        }
    }

    /// Bytes per element, for the kernel argument metadata and for deriving
    /// [`crate::gemm::GemmSpec::elem_bytes`].
    pub fn size(self) -> u32 {
        match self {
            MppDtype::BF16 | MppDtype::F16 => 2,
            MppDtype::F32 => 4,
        }
    }
}

/// A single-tile MPP GEMM: `C[M,N] = A[M,K] @ B[K,N]`, row-major, one `BM x BN`
/// output tile per threadgroup. Shapes are baked in, matching how every other
/// kernel here is compiled per shape.
#[derive(Debug, Clone)]
pub struct MppGemm {
    pub entry: String,
    pub dtype: MppDtype,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    /// Row strides in elements.
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub bm: i32,
    pub bn: i32,
    /// Destination element type. Differs from `dtype` for split-K, whose partials
    /// must be f32: the run symbol becomes `..._dv_b16_dv_b16_f32` and the store
    /// `..._store_dv_f32`.
    pub out_dtype: MppDtype,
    /// Number of K splits. 1 is a plain single-pass GEMM writing `C` directly.
    ///
    /// Above 1 the kernel takes the split index from `tgid.z`, restricts its A and B
    /// slices to K range `[s*K/splits, (s+1)*K/splits)`, and writes plane `s` of an
    /// `[splits, M, N]` f32 partial buffer for a reduce pass to sum. This exists
    /// because MPP alone is starved when M is small - at M=1 the grid is just
    /// `ceil(N/BN)` threadgroups and it runs 3-10x slower than a scalar split-K
    /// kernel. `splits` must divide K.
    pub splits: i32,
    /// B is `[N, K]` and the op computes `A @ B.t()`, the `Linear` weight layout.
    ///
    /// `matmul2d_descriptor` carries `transB` as one byte, so this costs nothing to
    /// express; whether the DRIVER honours it is what `tests/mpp_gemm.rs` checks. It
    /// matters because a prefill GEMM in this layout would otherwise need a bespoke
    /// single-SIMD-group SGMMA kernel, ~12% slower on the same cells.
    pub trans_b: bool,
    /// SIMD groups per threadgroup; the threadgroup must be `simdgroups * 32` wide.
    ///
    /// Use 1. The op reads the group count at runtime, so wider values are correct
    /// (covered by `tests/mpp_gemm.rs`), but MEASURED 6x slower: 156 ms against
    /// 24 ms at 4096x11008x4096 with 4 groups. One SIMD group per threadgroup is
    /// also the fastest config of everything tried, emitted or MSL-compiled.
    pub simdgroups: i32,
}

impl MppGemm {
    /// Threadgroup grid this kernel must be dispatched with, as `(x, y, z)`.
    /// x indexes N tiles and y indexes M tiles, matching the offsets below.
    pub fn grid(&self) -> (usize, usize, usize) {
        (
            (self.n as usize).div_ceil(self.bn as usize),
            (self.m as usize).div_ceil(self.bm as usize),
            self.splits.max(1) as usize,
        )
    }

    /// Elements in the f32 partial buffer a split-K launch writes (0 when
    /// `splits <= 1`, where the kernel writes `C` directly).
    pub fn partial_elems(&self) -> usize {
        if self.splits <= 1 {
            0
        } else {
            self.splits as usize * self.m as usize * self.n as usize
        }
    }

    /// Threads per threadgroup.
    pub fn threadgroup(&self) -> (usize, usize, usize) {
        (self.simdgroups as usize * 32, 1, 1)
    }

    /// This config's launch geometry, as one value.
    pub fn launch(&self) -> crate::Launch {
        crate::Launch::new(self.grid(), self.threadgroup())
    }
}

/// Emit a complete AIR module for `cfg`, ready for `xcrun metal-as`.
pub fn emit_mpp_gemm(cfg: &MppGemm, profile: &TargetProfile) -> String {
    let MppGemm {
        entry,
        dtype,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        bm,
        bn,
        ..
    } = cfg;
    let t = dtype.ll();
    let ot = cfg.out_dtype.ll();
    let sfx = dtype.suffix();
    let osfx = cfg.out_dtype.suffix();
    let tag = dtype.tag();
    let otag = cfg.out_dtype.tag();
    // (left, right, dest) for init/data_size/get_element_pointer; the store takes
    // only the two INPUT tags - read off the frontend, not inferred.
    let tags3 = format!("i32 {tag}, i32 {tag}, i32 {otag}");
    let tags2 = format!("i32 {tag}, i32 {tag}");
    let splits = cfg.splits.max(1);
    let ks = (k + splits - 1) / splits;
    // matmul2d_descriptor{BM, BN, BK, transA, transB, relaxed, mode}. BK is -1
    // (`dynamic_extent`): the op walks all of K itself.
    // The cooperative-destination helpers take the descriptor as UNPACKED SCALARS, and
    // every one of them (init / data_size / get_element_pointer / store) carries it. An
    // MSL round-trip of the cooperative NN vs NT pair shows `i8 0, i8 0, i8 1` becoming
    // `i8 0, i8 1, i8 1` in ALL FOUR. Setting only the descriptor global that `run`
    // memcpys leaves the destination tensor laid out for NN while the op computes NT -
    // which is exactly the bug this hunt chased.
    let tr = i32::from(cfg.trans_b);
    let desc_args = format!("i32 {bm}, i32 {bn}, i32 -1, i8 0, i8 {tr}, i8 1, i32 0");
    let mut o = String::new();

    o.push_str(&format!(
        "target datalayout = \"{}\"\n",
        profile.datalayout()
    ));
    o.push_str(&format!("target triple = \"{}\"\n\n", profile.triple()));

    o.push_str(
        "%\"struct.mpp::tensor_ops::matmul2d_descriptor\" = type { i32, i32, i32, i8, i8, i8, i32 }\n\
         %\"struct.metal::array\" = type { [2 x i32] }\n\
         %struct._tensor_t = type opaque\n\n",
    );
    o.push_str(&format!(
        "@mpp.desc = internal unnamed_addr constant \
         %\"struct.mpp::tensor_ops::matmul2d_descriptor\" \
         {{ i32 {bm}, i32 {bn}, i32 -1, i8 0, i8 {tr}, i8 1, i32 0 }}, align 4\n\n"
    ));

    // Tensor descriptor ABI (defined in the toolchain's tensor.metallib) ...
    o.push_str(
        "declare i16 @air.get_descriptor_size_tensor(i16, i16) local_unnamed_addr #1\n\
         declare i32 @air.get_simdgroup_size.i32() local_unnamed_addr #1\n\
         declare void @air.init_strided_private_tensor.i32.global(%struct._tensor_t* nocapture writeonly, i16, i8 addrspace(1)* readnone, i8* nocapture readonly, i8* nocapture readonly, i8) local_unnamed_addr #2\n\
         declare i32 @air.get_extent_private_tensor.i32(%struct._tensor_t* nocapture readonly, i16, i16) local_unnamed_addr #2\n\
         declare void @air.slice_private_tensor_private_tensor.s.i32(%struct._tensor_t* nocapture writeonly, %struct._tensor_t* nocapture readonly, i16, i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #2\n\
         declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #3\n\
         declare i32 @llvm.smin.i32(i32, i32) #5\n",
    );
    // ... and the matmul itself, resolved by the driver at pipeline creation.
    o.push_str(&format!(
        "declare i64 @__tensorops_impl_matmul2d_op_cooperative_destination_data_size(i32, i32, i32, i8, i8, i8, i32, i32 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #4 section \"air.externally_defined\"\n\
         declare void @__tensorops_impl_matmul2d_op_cooperative_destination_tensor_init(i32, i32, i32, i8, i8, i8, i32, i8* noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #4 section \"air.externally_defined\"\n\
         declare i8* @__tensorops_impl_matmul2d_op_cooperative_destination_tensor_get_element_pointer(i32, i32, i32, i8, i8, i8, i32, i8* noundef, i16 noundef zeroext, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #4 section \"air.externally_defined\"\n\
         declare void @__tensorops_impl_matmul2d_op_run_cooperative_dv_{sfx}_dv_{sfx}_{osfx}(%\"struct.mpp::tensor_ops::matmul2d_descriptor\"* noundef nonnull align 4 dereferenceable(20), i8* noundef, i32 noundef, i8* noundef, i32 noundef, i8* noundef, i32 noundef) local_unnamed_addr #4 section \"air.externally_defined\"\n\
         declare void @__tensorops_impl_matmul2d_op_cooperative_destination_tensor_store_dv_{osfx}(i32, i32, i32, i8, i8, i8, i32, i8* noundef, i8* noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #4 section \"air.externally_defined\"\n\n"
    ));

    o.push_str(&format!(
        "define void @{entry}({ot} addrspace(1)* noundef \"air-buffer-no-alias\" %c, \
         {t} addrspace(1)* noundef \"air-buffer-no-alias\" %a, \
         {t} addrspace(1)* noundef \"air-buffer-no-alias\" %b, \
         <3 x i32> noundef %tg) local_unnamed_addr #0 {{\n\
         entry:\n"
    ));

    // Descriptor sizes. Both are runtime queries: the tensor descriptor is sized by
    // rank and kind, the cooperative tensor by the matmul shape and SIMD width.
    o.push_str(
        "  %dsz16 = call i16 @air.get_descriptor_size_tensor(i16 2, i16 4)\n\
         \x20 %dsz = zext i16 %dsz16 to i64\n\
         \x20 %sg = call i32 @air.get_simdgroup_size.i32()\n",
    );
    o.push_str(&format!(
        "  %csz = call i64 @__tensorops_impl_matmul2d_op_cooperative_destination_data_size({desc_args}, {tags3}, i32 %sg)\n"
    ));
    for v in ["ta", "tb", "tc", "sa", "sb", "sc"] {
        o.push_str(&format!("  %{v} = alloca i8, i64 %dsz, align 8\n"));
    }
    o.push_str("  %coop = alloca i8, i64 %csz, align 8\n");

    // Whole-tensor descriptors. Extents are (columns, rows) - dim 0 is the
    // contiguous one - and strides are {1, row_stride}.
    let ext = |o: &mut String, name: &str, d0: String, d1: String| {
        o.push_str(&format!(
            "  %{name} = alloca %\"struct.metal::array\", align 4\n\
             \x20 %{name}.0 = getelementptr inbounds %\"struct.metal::array\", %\"struct.metal::array\"* %{name}, i64 0, i32 0, i64 0\n\
             \x20 store i32 {d0}, i32* %{name}.0, align 4\n\
             \x20 %{name}.1 = getelementptr inbounds %\"struct.metal::array\", %\"struct.metal::array\"* %{name}, i64 0, i32 0, i64 1\n\
             \x20 store i32 {d1}, i32* %{name}.1, align 4\n\
             \x20 %{name}.p = bitcast %\"struct.metal::array\"* %{name} to i8*\n"
        ));
    };
    ext(&mut o, "ea", k.to_string(), m.to_string());
    ext(&mut o, "sla", "1".into(), lda.to_string());
    // With `trans_b` the operand is [N, K]: the fast dim becomes K and the slow dim N,
    // so BOTH the extents and the slice origin below swap. Flipping only the descriptor
    // byte builds a working pipeline that computes the wrong thing (measured 6.3e-1 rel
    // err), which is why this is expressed here rather than left to the caller.
    if cfg.trans_b {
        ext(&mut o, "eb", k.to_string(), n.to_string());
    } else {
        ext(&mut o, "eb", n.to_string(), k.to_string());
    }
    ext(&mut o, "slb", "1".into(), ldb.to_string());
    ext(&mut o, "ec", n.to_string(), m.to_string());
    ext(&mut o, "slc", "1".into(), ldc.to_string());

    // Split index and this split's K window.
    if splits > 1 {
        // Window is ceil(K/splits) and the last split takes what is left, so K need
        // not divide evenly.
        o.push_str(&format!(
            "  %s = extractelement <3 x i32> %tg, i64 2\n\
             \x20 %k0 = mul nsw i32 %s, {ks}\n\
             \x20 %krem = sub nsw i32 {k}, %k0\n\
             \x20 %kext = call i32 @llvm.smin.i32(i32 {ks}, i32 %krem)\n\
             \x20 %plane = mul nsw i32 %s, {}\n\
             \x20 %cbase = getelementptr {ot}, {ot} addrspace(1)* %c, i32 %plane\n",
            m * n
        ));
    } else {
        o.push_str(&format!(
            "  %cbase = getelementptr {ot}, {ot} addrspace(1)* %c, i32 0\n"
        ));
    }
    for (t_ptr, buf, e, s, ty) in [
        ("ta", "%a", "ea", "sla", t),
        ("tb", "%b", "eb", "slb", t),
        ("tc", "%cbase", "ec", "slc", ot),
    ] {
        let name = buf.trim_start_matches('%');
        o.push_str(&format!(
            "  %{t_ptr}.t = bitcast i8* %{t_ptr} to %struct._tensor_t*\n\
             \x20 %{name}.i8 = bitcast {ty} addrspace(1)* {buf} to i8 addrspace(1)*\n\
             \x20 call void @air.init_strided_private_tensor.i32.global(%struct._tensor_t* %{t_ptr}.t, i16 2, i8 addrspace(1)* %{name}.i8, i8* %{e}.p, i8* %{s}.p, i8 2)\n"
        ));
    }

    // Tile origins: x indexes N, y indexes M.
    o.push_str(&format!(
        "  %tgy = extractelement <3 x i32> %tg, i64 1\n\
         \x20 %m_off = mul nsw i32 %tgy, {bm}\n\
         \x20 %tgx = extractelement <3 x i32> %tg, i64 0\n\
         \x20 %n_off = mul nsw i32 %tgx, {bn}\n"
    ));

    // Slices. Extents are what remains from the origin to the end of the tensor;
    // the op clamps to its own BM/BN, so edge tiles need no special case.
    // A is sliced on K (dim 0) and M (dim 1), B on N (dim 0) and K (dim 1). For
    // split-K the K origin is this split's `k0` and the K extent is exactly `KS`,
    // which is what confines the op's internal K loop to this split's window.
    let (ka_o, ka_e) = if splits > 1 {
        ("%k0".to_string(), "%kext".to_string())
    } else {
        ("0".to_string(), String::new())
    };
    // `shrink0`/`shrink1` name the OFFSET to subtract from that dim's extent, rather
    // than a bool - the offset is not a function of the dimension index. B's N axis is
    // dim 0 when `trans_b` is false and dim 1 when it is true, and hardcoding
    // "dim0 shrinks by n_off, dim1 by m_off" is what made the first transB attempt
    // compute garbage (5.5e0 rel err): it clipped B's N extent by the M offset.
    for (dst, src, o0, o1, shrink0, shrink1) in [
        ("sa", "ta", ka_o.as_str(), "%m_off", None, Some("%m_off")),
        if cfg.trans_b {
            ("sb", "tb", ka_o.as_str(), "%n_off", None, Some("%n_off"))
        } else {
            ("sb", "tb", "%n_off", ka_o.as_str(), Some("%n_off"), None)
        },
        (
            "sc",
            "tc",
            "%n_off",
            "%m_off",
            Some("%n_off"),
            Some("%m_off"),
        ),
    ] {
        o.push_str(&format!(
            "  %{dst}.o = alloca %\"struct.metal::array\", align 4\n\
             \x20 %{dst}.o0 = getelementptr inbounds %\"struct.metal::array\", %\"struct.metal::array\"* %{dst}.o, i64 0, i32 0, i64 0\n\
             \x20 store i32 {o0}, i32* %{dst}.o0, align 4\n\
             \x20 %{dst}.o1 = getelementptr inbounds %\"struct.metal::array\", %\"struct.metal::array\"* %{dst}.o, i64 0, i32 0, i64 1\n\
             \x20 store i32 {o1}, i32* %{dst}.o1, align 4\n\
             \x20 %{dst}.op = bitcast %\"struct.metal::array\"* %{dst}.o to i8*\n\
             \x20 %{dst}.x0 = call i32 @air.get_extent_private_tensor.i32(%struct._tensor_t* %{src}.t, i16 2, i16 0)\n\
             \x20 %{dst}.x1 = call i32 @air.get_extent_private_tensor.i32(%struct._tensor_t* %{src}.t, i16 2, i16 1)\n"
        ));
        // K axis: A's dim 0 always; B's dim 1 normally, dim 0 when transposed.
        let k_dim0 = (dst == "sa") || (dst == "sb" && cfg.trans_b);
        let k_dim1 = dst == "sb" && !cfg.trans_b;
        let e0 = if let Some(off) = shrink0 {
            o.push_str(&format!("  %{dst}.e0 = sub i32 %{dst}.x0, {off}\n"));
            format!("%{dst}.e0")
        } else if k_dim0 && !ka_e.is_empty() {
            ka_e.clone() // exactly this split's K window
        } else {
            format!("%{dst}.x0")
        };
        let e1 = if let Some(off) = shrink1 {
            o.push_str(&format!("  %{dst}.e1 = sub i32 %{dst}.x1, {off}\n"));
            format!("%{dst}.e1")
        } else if k_dim1 && !ka_e.is_empty() {
            ka_e.clone()
        } else {
            format!("%{dst}.x1")
        };
        o.push_str(&format!(
            "  %{dst}.e = alloca %\"struct.metal::array\", align 4\n\
             \x20 %{dst}.ep0 = getelementptr inbounds %\"struct.metal::array\", %\"struct.metal::array\"* %{dst}.e, i64 0, i32 0, i64 0\n\
             \x20 store i32 {e0}, i32* %{dst}.ep0, align 4\n\
             \x20 %{dst}.ep1 = getelementptr inbounds %\"struct.metal::array\", %\"struct.metal::array\"* %{dst}.e, i64 0, i32 0, i64 1\n\
             \x20 store i32 {e1}, i32* %{dst}.ep1, align 4\n\
             \x20 %{dst}.epp = bitcast %\"struct.metal::array\"* %{dst}.e to i8*\n\
             \x20 %{dst}.t = bitcast i8* %{dst} to %struct._tensor_t*\n\
             \x20 call void @air.slice_private_tensor_private_tensor.s.i32(%struct._tensor_t* %{dst}.t, %struct._tensor_t* %{src}.t, i16 2, i8* %{dst}.op, i8* %{dst}.epp)\n"
        ));
    }

    // Cooperative destination tensor, then the matmul, then the store.
    o.push_str(&format!(
        "  call void @__tensorops_impl_matmul2d_op_cooperative_destination_tensor_init({desc_args}, i8* %coop, {tags3}, i32 %sg)\n\
         \x20 %desc = alloca %\"struct.mpp::tensor_ops::matmul2d_descriptor\", align 4\n\
         \x20 %desc.i8 = bitcast %\"struct.mpp::tensor_ops::matmul2d_descriptor\"* %desc to i8*\n\
         \x20 call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %desc.i8, i8* align 4 bitcast (%\"struct.mpp::tensor_ops::matmul2d_descriptor\"* @mpp.desc to i8*), i64 20, i1 false)\n\
         \x20 %dst = call i8* @__tensorops_impl_matmul2d_op_cooperative_destination_tensor_get_element_pointer({desc_args}, i8* %coop, i16 zeroext -1, {tags3})\n\
         \x20 call void @__tensorops_impl_matmul2d_op_run_cooperative_dv_{sfx}_dv_{sfx}_{osfx}(%\"struct.mpp::tensor_ops::matmul2d_descriptor\"* %desc, i8* %sa, i32 2, i8* %sb, i32 2, i8* %dst, i32 %sg)\n\
         \x20 call void @__tensorops_impl_matmul2d_op_cooperative_destination_tensor_store_dv_{osfx}({desc_args}, i8* %coop, i8* %sc, i32 2, {tags2}, i32 %sg)\n\
         \x20 ret void\n}}\n\n"
    ));

    o.push_str(
        "attributes #0 = { convergent nounwind \"frame-pointer\"=\"all\" \"no-builtins\" \"no-trapping-math\"=\"true\" }\n\
         attributes #1 = { mustprogress nofree nosync nounwind readnone willreturn }\n\
         attributes #2 = { argmemonly mustprogress nounwind willreturn }\n\
         attributes #3 = { argmemonly mustprogress nofree nounwind willreturn }\n\
         attributes #4 = { convergent nobuiltin nounwind \"no-builtins\" }\n\
         attributes #5 = { nounwind readnone speculatable willreturn }\n\n",
    );

    let esz = dtype.size();
    let oesz = cfg.out_dtype.size();
    o.push_str(&format!(
        "!llvm.module.flags = !{{!0, !1}}\n\
         !air.kernel = !{{!2}}\n\
         !air.version = !{{!8}}\n\
         !air.language_version = !{{!9}}\n\
         !llvm.ident = !{{!10}}\n\n\
         !0 = !{{i32 2, !\"SDK Version\", [2 x i32] [i32 26, i32 0]}}\n\
         !1 = !{{i32 7, !\"air.max_device_buffers\", i32 31}}\n\
         !2 = !{{void ({ot} addrspace(1)*, {t} addrspace(1)*, {t} addrspace(1)*, <3 x i32>)* @{entry}, !3, !4}}\n\
         !3 = !{{}}\n\
         !4 = !{{!5, !6, !7, !11}}\n\
         !5 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {oesz}, !\"air.arg_type_align_size\", i32 {oesz}, !\"air.arg_type_name\", !\"{ot}\", !\"air.arg_name\", !\"c\"}}\n\
         !6 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {esz}, !\"air.arg_type_align_size\", i32 {esz}, !\"air.arg_type_name\", !\"{t}\", !\"air.arg_name\", !\"a\"}}\n\
         !7 = !{{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 {esz}, !\"air.arg_type_align_size\", i32 {esz}, !\"air.arg_type_name\", !\"{t}\", !\"air.arg_name\", !\"b\"}}\n\
         !8 = !{{i32 2, i32 8, i32 0}}\n\
         !9 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n\
         !10 = !{{!\"wax-metal\"}}\n\
         !11 = !{{i32 3, !\"air.threadgroup_position_in_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"tg\"}}\n"
    ));
    o
}

// ---------------------------------------------------------------------------
// In-process bitcode emission
// ---------------------------------------------------------------------------
//
// The same kernel as `emit_mpp_gemm`, built straight into LLVM bitcode so MPP no
// longer needs `xcrun metal-as`. Three simplifications versus the text form, each
// removing a construct the bitcode writer would otherwise have to model:
//
//   * `%struct._tensor_t*` and the descriptor pointers become plain `i8*`. Every
//     one of them is bitcast to `i8*` in the text version anyway, and the callees
//     are resolved by NAME at pipeline-creation time, so nothing type-checks the
//     pointee. This drops named and opaque struct types entirely.
//   * `%"struct.metal::array"` (a `{ [2 x i32] }` wrapper) becomes a bare
//     `[2 x i32]`, which the type table already supports. Same address, one fewer
//     GEP index.
//   * the descriptor is built by storing five i32 words into an alloca rather than
//     `memcpy`ing from a module constant. `{i32,i32,i32,i8,i8,i8,i32}` at align 4
//     is 20 bytes whose bytes 12..16 are `{0,0,1,pad}` = 0x00010000, so the whole
//     thing is five plain word stores. This drops both the global initializer and
//     the memcpy intrinsic.
//
// `alloca`'s count operand is i32 in this encoder, so the runtime descriptor sizes
// are zext/trunc'ed to i32 rather than the i64 the text form uses.

use crate::bitcode::func::{Body, Inst, Val, cast};
use crate::bitcode::metadata::{Access, KernelArg, KernelMetadata};
use crate::bitcode::{Module as BcModule, Ty};

/// Slot numbers of the module-scope declarations, in declaration order.
struct Decls {
    desc_size: u32,
    simd_size: u32,
    init_tensor: u32,
    get_extent: u32,
    slice_tensor: u32,
    smin: u32,
    coop_size: u32,
    coop_init: u32,
    coop_elem_ptr: u32,
    run: u32,
    store: u32,
    entry: u32,
}

/// Lower `cfg` to a kernel bundled with its launch geometry and scratch requirement.
///
/// Prefer this over [`emit_mpp_gemm_metallib`]: MPP's grid is `(N/BN, M/BM, splits)`
/// with a 32-wide threadgroup and split-K needs an `[S,M,N]` f32 partial buffer, none
/// of which a caller can infer from the shape. Returning them together is what makes
/// it safe to choose this backend over SGMMA without every call site having to know
/// which one it got - see [`crate::launch`].
pub fn lower_mpp_gemm(cfg: &MppGemm, profile: &TargetProfile) -> crate::LoweredKernel {
    crate::LoweredKernel {
        metallib: emit_mpp_gemm_metallib(cfg, profile),
        entry: cfg.entry.clone(),
        launch: cfg.launch(),
        // Split-K writes f32 partials for a reduce pass; a single-pass GEMM writes
        // the destination directly and needs nothing.
        scratch_bytes: cfg.partial_elems() * 4,
        scalar_args: Vec::new(),
    }
}

/// Emit a complete `.metallib` for `cfg` with no external tools.
///
/// Byte-for-byte equivalence with [`emit_mpp_gemm`] is not the goal - equivalent GPU
/// results are, which `wax-metal benches/tests/mpp_gemm.rs` checks by running both
/// paths against the same CPU reference.
pub fn emit_mpp_gemm_metallib(cfg: &MppGemm, profile: &TargetProfile) -> Vec<u8> {
    let bc = mpp_bitcode(cfg, profile);
    crate::metallib::wrap(
        &bc,
        &[crate::metallib::Function::kernel(&cfg.entry)],
        profile,
    )
}

/// The raw LLVM bitcode, split out so it can be dumped and read with `llvm-dis`.
pub fn mpp_bitcode(cfg: &MppGemm, profile: &TargetProfile) -> Vec<u8> {
    let (i8t, i16t, i32t, i64t) = (Ty::Int(8), Ty::Int(16), Ty::Int(32), Ty::Int(64));
    let p0i8 = Ty::ptr(i8t.clone(), 0);
    let p1i8 = Ty::ptr(i8t.clone(), 1);
    let arr2 = Ty::Array(2, Box::new(i32t.clone()));
    let v3i32 = Ty::vec(3, i32t.clone());
    let elem = |d: MppDtype| match d {
        MppDtype::BF16 => Ty::BFloat,
        MppDtype::F16 => Ty::Half,
        MppDtype::F32 => Ty::Float,
    };
    let p1e = Ty::ptr(elem(cfg.dtype), 1);
    let p1oe = Ty::ptr(elem(cfg.out_dtype), 1);

    let mut m = BcModule::new(profile.triple(), profile.datalayout());
    let ia = m.intrinsic_attrs();
    let ka = m.kernel_attrs();

    // Declaration order fixes the call records' callee ids, so it must not shift.
    let mut next = 0u32;
    let mut slot = || {
        let s = next;
        next += 1;
        s
    };
    let d = Decls {
        desc_size: slot(),
        simd_size: slot(),
        init_tensor: slot(),
        get_extent: slot(),
        slice_tensor: slot(),
        smin: slot(),
        coop_size: slot(),
        coop_init: slot(),
        coop_elem_ptr: slot(),
        run: slot(),
        store: slot(),
        entry: slot(),
    };

    // The tensor descriptor ABI, defined in the toolchain's tensor.metallib.
    let t_desc_size = Ty::func(i16t.clone(), vec![i16t.clone(), i16t.clone()]);
    let t_simd_size = Ty::func(i32t.clone(), vec![]);
    let t_init = Ty::func(
        Ty::Void,
        vec![
            p0i8.clone(),
            i16t.clone(),
            p1i8.clone(),
            p0i8.clone(),
            p0i8.clone(),
            i8t.clone(),
        ],
    );
    let t_extent = Ty::func(i32t.clone(), vec![p0i8.clone(), i16t.clone(), i16t.clone()]);
    let t_slice = Ty::func(
        Ty::Void,
        vec![
            p0i8.clone(),
            p0i8.clone(),
            i16t.clone(),
            p0i8.clone(),
            p0i8.clone(),
        ],
    );
    let t_smin = Ty::func(i32t.clone(), vec![i32t.clone(), i32t.clone()]);
    m.declare(
        "air.get_descriptor_size_tensor",
        t_desc_size.clone(),
        true,
        ia,
    );
    m.declare("air.get_simdgroup_size.i32", t_simd_size.clone(), true, ia);
    m.declare(
        "air.init_strided_private_tensor.i32.global",
        t_init.clone(),
        true,
        ia,
    );
    m.declare(
        "air.get_extent_private_tensor.i32",
        t_extent.clone(),
        true,
        ia,
    );
    m.declare(
        "air.slice_private_tensor_private_tensor.s.i32",
        t_slice.clone(),
        true,
        ia,
    );
    m.declare("llvm.smin.i32", t_smin.clone(), true, ia);

    // The matmul itself. `air.externally_defined` is what tells the driver to supply
    // these bodies at pipeline creation; nothing links them here.
    let dfields = || {
        vec![
            i32t.clone(),
            i32t.clone(),
            i32t.clone(),
            i8t.clone(),
            i8t.clone(),
            i8t.clone(),
            i32t.clone(),
        ]
    };
    let mut coop_size_p = dfields();
    coop_size_p.extend([i32t.clone(), i32t.clone(), i32t.clone(), i32t.clone()]);
    let mut coop_init_p = dfields();
    coop_init_p.extend([
        p0i8.clone(),
        i32t.clone(),
        i32t.clone(),
        i32t.clone(),
        i32t.clone(),
    ]);
    let mut coop_ep_p = dfields();
    coop_ep_p.extend([
        p0i8.clone(),
        i16t.clone(),
        i32t.clone(),
        i32t.clone(),
        i32t.clone(),
    ]);
    let run_p = vec![
        p0i8.clone(),
        p0i8.clone(),
        i32t.clone(),
        p0i8.clone(),
        i32t.clone(),
        p0i8.clone(),
        i32t.clone(),
    ];
    let mut store_p = dfields();
    store_p.extend([
        p0i8.clone(),
        p0i8.clone(),
        i32t.clone(),
        i32t.clone(),
        i32t.clone(),
        i32t.clone(),
    ]);

    let t_coop_size = Ty::func(i64t.clone(), coop_size_p);
    let t_coop_init = Ty::func(Ty::Void, coop_init_p);
    let t_coop_ep = Ty::func(p0i8.clone(), coop_ep_p);
    let t_run = Ty::func(Ty::Void, run_p);
    let t_store = Ty::func(Ty::Void, store_p);

    let sfx = cfg.dtype.suffix();
    let osfx = cfg.out_dtype.suffix();
    const SEC: &str = "air.externally_defined";
    for (name, ty) in [
        (
            "__tensorops_impl_matmul2d_op_cooperative_destination_data_size".to_string(),
            t_coop_size.clone(),
        ),
        (
            "__tensorops_impl_matmul2d_op_cooperative_destination_tensor_init".to_string(),
            t_coop_init.clone(),
        ),
        (
            "__tensorops_impl_matmul2d_op_cooperative_destination_tensor_get_element_pointer"
                .to_string(),
            t_coop_ep.clone(),
        ),
        (
            format!("__tensorops_impl_matmul2d_op_run_cooperative_dv_{sfx}_dv_{sfx}_{osfx}"),
            t_run.clone(),
        ),
        (
            format!("__tensorops_impl_matmul2d_op_cooperative_destination_tensor_store_dv_{osfx}"),
            t_store.clone(),
        ),
    ] {
        m.declare_in_section(&name, ty, true, ia, SEC);
    }

    let kty = Ty::func(
        Ty::Void,
        vec![p1oe.clone(), p1e.clone(), p1e.clone(), v3i32.clone()],
    );
    m.declare(&cfg.entry, kty.clone(), false, ka);
    m.types.intern(&i64t);
    m.types.intern(&arr2);

    let body = mpp_body(
        cfg,
        &d,
        &BodyTys {
            i8t: i8t.clone(),
            i16t: i16t.clone(),
            i32t: i32t.clone(),
            p0i8: p0i8.clone(),
            p1i8: p1i8.clone(),
            arr2: arr2.clone(),
            oelem: elem(cfg.out_dtype),
            t_desc_size,
            t_simd_size,
            t_init,
            t_extent,
            t_slice,
            t_smin,
            t_coop_size,
            t_coop_init,
            t_coop_ep,
            t_run,
            t_store,
        },
    );
    m.set_body(d.entry as usize, body);

    let esz = cfg.dtype.size();
    let oesz = cfg.out_dtype.size();
    let args = vec![
        KernelArg::Buffer {
            index: 0,
            address_space: 1,
            access: Access::ReadWrite,
            elem_size: oesz,
            elem_align: oesz,
            type_name: cfg.out_dtype.ll().to_string(),
            name: "c".into(),
        },
        KernelArg::Buffer {
            index: 1,
            address_space: 1,
            access: Access::Read,
            elem_size: esz,
            elem_align: esz,
            type_name: cfg.dtype.ll().to_string(),
            name: "a".into(),
        },
        KernelArg::Buffer {
            index: 2,
            address_space: 1,
            access: Access::Read,
            elem_size: esz,
            elem_align: esz,
            type_name: cfg.dtype.ll().to_string(),
            name: "b".into(),
        },
        KernelArg::Builtin {
            kind: "air.threadgroup_position_in_grid".into(),
            type_name: "uint3".into(),
            name: "tg".into(),
        },
    ];

    let const_base = m.const_base();
    let km = KernelMetadata {
        entry_value_id: m.func_value_id(d.entry),
        entry_fn_ptr_ty: Ty::ptr(kty, 0),
        args: &args,
        air_version: profile.air_version,
        language_version: profile.language_version,
        limits: profile.limits,
        producer: "wax-metal",
    };
    let mut md = std::mem::take(&mut m.md);
    let mut consts = std::mem::take(&mut m.consts);
    let mut types = std::mem::take(&mut m.types);
    km.build(&mut md, &mut consts, &mut types, const_base);
    types.intern(&km.entry_fn_ptr_ty);
    m.md = md;
    m.consts = consts;
    m.types = types;
    m.to_bitcode()
}

/// Types the body builder needs, gathered so the signature stays readable.
struct BodyTys {
    i8t: Ty,
    i16t: Ty,
    i32t: Ty,
    p0i8: Ty,
    p1i8: Ty,
    arr2: Ty,
    oelem: Ty,
    t_desc_size: Ty,
    t_simd_size: Ty,
    t_init: Ty,
    t_extent: Ty,
    t_slice: Ty,
    t_smin: Ty,
    t_coop_size: Ty,
    t_coop_init: Ty,
    t_coop_ep: Ty,
    t_run: Ty,
    t_store: Ty,
}

fn mpp_body(cfg: &MppGemm, d: &Decls, t: &BodyTys) -> Body {
    let mut b = Body::new(4);
    let (c_arg, a_arg, b_arg, tg_arg) = (Val::Arg(0), Val::Arg(1), Val::Arg(2), Val::Arg(3));

    let i32c = |b: &mut Body, v: i32| b.iconst(t.i32t.clone(), v as i64);
    let i16c = |b: &mut Body, v: i32| b.iconst(t.i16t.clone(), v as i64);
    let i8c = |b: &mut Body, v: i32| b.iconst(t.i8t.clone(), v as i64);

    let splits = cfg.splits.max(1);
    let ks = (cfg.k + splits - 1) / splits;

    // matmul2d_descriptor{BM, BN, BK, transA, transB, relaxed, mode}; BK is -1
    // (`dynamic_extent`), so the op walks all of K itself.
    // Same unpacked-scalar descriptor as the text emitter - transB must appear in EVERY
    // cooperative-destination helper, not just the `run` descriptor.
    let desc_args = |b: &mut Body| {
        vec![
            i32c(b, cfg.bm),
            i32c(b, cfg.bn),
            i32c(b, -1),
            i8c(b, 0),
            i8c(b, i32::from(cfg.trans_b)),
            i8c(b, 1),
            i32c(b, 0),
        ]
    };
    let tag = cfg.dtype.tag();
    let otag = cfg.out_dtype.tag();

    // Descriptor sizes: both are runtime queries.
    let two16 = i16c(&mut b, 2);
    let four16 = i16c(&mut b, 4);
    let dsz16 = b.push(Inst::Call {
        fnty: t.t_desc_size.clone(),
        callee: Val::Func(d.desc_size),
        args: vec![two16.clone(), four16],
        returns_value: true,
        fast: false,
    });
    // The encoder's alloca takes an i32 count, so widen to i32 rather than the i64
    // the text form uses; the value is identical.
    let dsz = b.push(Inst::Cast {
        op: cast::ZEXT,
        val: dsz16,
        to: t.i32t.clone(),
    });
    let sg = b.push(Inst::Call {
        fnty: t.t_simd_size.clone(),
        callee: Val::Func(d.simd_size),
        args: vec![],
        returns_value: true,
        fast: false,
    });

    let mut csz_args = desc_args(&mut b);
    let tag_c = i32c(&mut b, tag);
    let otag_c = i32c(&mut b, otag);
    csz_args.extend([tag_c.clone(), tag_c.clone(), otag_c.clone(), sg.clone()]);
    let csz64 = b.push(Inst::Call {
        fnty: t.t_coop_size.clone(),
        callee: Val::Func(d.coop_size),
        args: csz_args,
        returns_value: true,
        fast: false,
    });
    let csz = b.push(Inst::Cast {
        op: cast::TRUNC,
        val: csz64,
        to: t.i32t.clone(),
    });

    // Six tensor descriptors (3 whole, 3 sliced) plus the cooperative destination.
    let mut tensors = Vec::new();
    for _ in 0..6 {
        tensors.push(b.push(Inst::Alloca {
            ty: t.i8t.clone(),
            count: dsz.clone(),
            align: 8,
        }));
    }
    let (ta, tb, tc) = (tensors[0].clone(), tensors[1].clone(), tensors[2].clone());
    let (sa, sb, sc) = (tensors[3].clone(), tensors[4].clone(), tensors[5].clone());
    let coop = b.push(Inst::Alloca {
        ty: t.i8t.clone(),
        count: csz,
        align: 8,
    });

    // `[2 x i32]` pair on the stack, returned as `i8*`.
    let mk_pair = |b: &mut Body, d0: Val, d1: Val| -> Val {
        let one = i32c(b, 1);
        let zero = i32c(b, 0);
        let p = b.push(Inst::Alloca {
            ty: t.arr2.clone(),
            count: one.clone(),
            align: 4,
        });
        let g0 = b.push(Inst::Gep {
            inbounds: true,
            elem: t.arr2.clone(),
            ptr: p.clone(),
            indices: vec![zero.clone(), zero.clone()],
        });
        b.push(Inst::Store {
            ptr: g0,
            val: d0,
            align: 4,
        });
        let g1 = b.push(Inst::Gep {
            inbounds: true,
            elem: t.arr2.clone(),
            ptr: p.clone(),
            indices: vec![zero, one],
        });
        b.push(Inst::Store {
            ptr: g1,
            val: d1,
            align: 4,
        });
        b.push(Inst::Cast {
            op: cast::BITCAST,
            val: p,
            to: t.p0i8.clone(),
        })
    };

    // Whole-tensor extents are (columns, rows) - dim 0 is contiguous - with strides
    let one_c = i32c(&mut b, 1);
    let (kk, mm, nn) = (cfg.k, cfg.m, cfg.n);
    let d_k = i32c(&mut b, kk);
    let d_m = i32c(&mut b, mm);
    let d_n = i32c(&mut b, nn);
    let d_lda = i32c(&mut b, cfg.lda);
    let d_ldb = i32c(&mut b, cfg.ldb);
    let d_ldc = i32c(&mut b, cfg.ldc);
    let ea = mk_pair(&mut b, d_k.clone(), d_m.clone());
    let sla = mk_pair(&mut b, one_c.clone(), d_lda);
    // transB: B is [N, K], so its fast dim is K and its slow dim N (see the header's
    // "K = B.extent(0) for NT" rule).
    let eb = if cfg.trans_b {
        mk_pair(&mut b, d_k.clone(), d_n.clone())
    } else {
        mk_pair(&mut b, d_n.clone(), d_k.clone())
    };
    let slb = mk_pair(&mut b, one_c.clone(), d_ldb);
    let ec = mk_pair(&mut b, d_n.clone(), d_m.clone());
    let slc = mk_pair(&mut b, one_c.clone(), d_ldc);

    // Split index and this split's K window. `ceil(K/splits)` with the last split
    // clamped, so K need not divide evenly.
    let (k_origin, k_extent, cbase) = if splits > 1 {
        let two_idx = i32c(&mut b, 2);
        let s = b.push(Inst::ExtractElt {
            vec: tg_arg.clone(),
            idx: two_idx,
        });
        let ks_c = i32c(&mut b, ks);
        let k0 = b.push(Inst::BinOp {
            op: crate::bitcode::func::binop::MUL,
            lhs: s.clone(),
            rhs: ks_c.clone(),
        });
        let k_c = i32c(&mut b, kk);
        let krem = b.push(Inst::BinOp {
            op: crate::bitcode::func::binop::SUB,
            lhs: k_c,
            rhs: k0.clone(),
        });
        let kext = b.push(Inst::Call {
            fnty: t.t_smin.clone(),
            callee: Val::Func(d.smin),
            args: vec![ks_c, krem],
            returns_value: true,
            fast: false,
        });
        let mn = i32c(&mut b, mm * nn);
        let plane = b.push(Inst::BinOp {
            op: crate::bitcode::func::binop::MUL,
            lhs: s,
            rhs: mn,
        });
        let cb = b.push(Inst::Gep {
            inbounds: false,
            elem: t.oelem.clone(),
            ptr: c_arg.clone(),
            indices: vec![plane],
        });
        (Some(k0), Some(kext), cb)
    } else {
        (None, None, c_arg.clone())
    };

    // Initialise the three whole-tensor descriptors.
    let two8 = i8c(&mut b, 2);
    for (tp, buf, e, s) in [
        (&ta, &a_arg, &ea, &sla),
        (&tb, &b_arg, &eb, &slb),
        (&tc, &cbase, &ec, &slc),
    ] {
        let i8p = b.push(Inst::Cast {
            op: cast::BITCAST,
            val: buf.clone(),
            to: t.p1i8.clone(),
        });
        let two16b = i16c(&mut b, 2);
        b.push(Inst::Call {
            fnty: t.t_init.clone(),
            callee: Val::Func(d.init_tensor),
            args: vec![tp.clone(), two16b, i8p, e.clone(), s.clone(), two8.clone()],
            returns_value: false,
            fast: false,
        });
    }

    // Tile origins: grid x indexes N, y indexes M.
    let zero_idx = i32c(&mut b, 0);
    let one_idx = i32c(&mut b, 1);
    let tgy = b.push(Inst::ExtractElt {
        vec: tg_arg.clone(),
        idx: one_idx,
    });
    let bm_c = i32c(&mut b, cfg.bm);
    let m_off = b.push(Inst::BinOp {
        op: crate::bitcode::func::binop::MUL,
        lhs: tgy,
        rhs: bm_c,
    });
    let tgx = b.push(Inst::ExtractElt {
        vec: tg_arg,
        idx: zero_idx,
    });
    let bn_c = i32c(&mut b, cfg.bn);
    let n_off = b.push(Inst::BinOp {
        op: crate::bitcode::func::binop::MUL,
        lhs: tgx,
        rhs: bn_c,
    });

    // Slices. Extents run from the origin to the end of the tensor; the op clamps to
    // its own BM/BN, so edge tiles need no special case. A is sliced on K (dim 0) and
    // M (dim 1), B on N (dim 0) and K (dim 1).
    let zero_o = i32c(&mut b, 0);
    let k_o = k_origin.clone().unwrap_or_else(|| zero_o.clone());
    // `shrink0`/`shrink1` carry WHICH offset to subtract, not a bool: B's N axis is dim 0
    // normally and dim 1 under transB, so "dim0 shrinks by n_off" is not an invariant.
    // `k_dim` is the dimension holding B's K axis, which also moves under transB.
    let (b_o0, b_o1, b_s0, b_s1, b_kdim) = if cfg.trans_b {
        (
            k_o.clone(),
            n_off.clone(),
            None,
            Some(n_off.clone()),
            0usize,
        )
    } else {
        (n_off.clone(), k_o.clone(), Some(n_off.clone()), None, 1)
    };
    for (dst, src, o0, o1, shrink0, shrink1, split_dim) in [
        (
            &sa,
            &ta,
            k_o.clone(),
            m_off.clone(),
            None,
            Some(m_off.clone()),
            0usize,
        ),
        (&sb, &tb, b_o0, b_o1, b_s0, b_s1, b_kdim),
        (
            &sc,
            &tc,
            n_off.clone(),
            m_off.clone(),
            Some(n_off.clone()),
            Some(m_off.clone()),
            2,
        ),
    ] {
        let origin = mk_pair(&mut b, o0, o1);
        let two16b = i16c(&mut b, 2);
        let zero16 = i16c(&mut b, 0);
        let one16 = i16c(&mut b, 1);
        let x0 = b.push(Inst::Call {
            fnty: t.t_extent.clone(),
            callee: Val::Func(d.get_extent),
            args: vec![src.clone(), two16b.clone(), zero16],
            returns_value: true,
            fast: false,
        });
        let x1 = b.push(Inst::Call {
            fnty: t.t_extent.clone(),
            callee: Val::Func(d.get_extent),
            args: vec![src.clone(), two16b.clone(), one16],
            returns_value: true,
            fast: false,
        });
        let e0 = if let Some(off) = shrink0 {
            b.push(Inst::BinOp {
                op: crate::bitcode::func::binop::SUB,
                lhs: x0,
                rhs: off,
            })
        } else if split_dim == 0 && k_extent.is_some() {
            k_extent.clone().expect("checked")
        } else {
            x0
        };
        let e1 = if let Some(off) = shrink1 {
            b.push(Inst::BinOp {
                op: crate::bitcode::func::binop::SUB,
                lhs: x1,
                rhs: off,
            })
        } else if split_dim == 1 && k_extent.is_some() {
            k_extent.clone().expect("checked")
        } else {
            x1
        };
        let extent = mk_pair(&mut b, e0, e1);
        b.push(Inst::Call {
            fnty: t.t_slice.clone(),
            callee: Val::Func(d.slice_tensor),
            args: vec![dst.clone(), src.clone(), two16b, origin, extent],
            returns_value: false,
            fast: false,
        });
    }

    // Cooperative destination tensor, the matmul, then the store.
    let mut init_args = desc_args(&mut b);
    init_args.extend([
        coop.clone(),
        tag_c.clone(),
        tag_c.clone(),
        otag_c.clone(),
        sg.clone(),
    ]);
    b.push(Inst::Call {
        fnty: t.t_coop_init.clone(),
        callee: Val::Func(d.coop_init),
        args: init_args,
        returns_value: false,
        fast: false,
    });

    // The 20-byte descriptor, as five i32 words. Bytes 12..16 are
    let one_a = i32c(&mut b, 1);
    let desc = b.push(Inst::Alloca {
        ty: Ty::Array(5, Box::new(t.i32t.clone())),
        count: one_a,
        align: 4,
    });
    let words = [
        cfg.bm,
        cfg.bn,
        -1,
        0x0001_0000 | (i32::from(cfg.trans_b) << 8),
        0,
    ];
    for (i, wv) in words.iter().enumerate() {
        let zero_g = i32c(&mut b, 0);
        let idx = i32c(&mut b, i as i32);
        let g = b.push(Inst::Gep {
            inbounds: true,
            elem: Ty::Array(5, Box::new(t.i32t.clone())),
            ptr: desc.clone(),
            indices: vec![zero_g, idx],
        });
        let v = i32c(&mut b, *wv);
        b.push(Inst::Store {
            ptr: g,
            val: v,
            align: 4,
        });
    }
    let desc_i8 = b.push(Inst::Cast {
        op: cast::BITCAST,
        val: desc,
        to: t.p0i8.clone(),
    });

    let mut ep_args = desc_args(&mut b);
    let neg1_16 = i16c(&mut b, -1);
    ep_args.extend([
        coop.clone(),
        neg1_16,
        tag_c.clone(),
        tag_c.clone(),
        otag_c.clone(),
    ]);
    let dst_ptr = b.push(Inst::Call {
        fnty: t.t_coop_ep.clone(),
        callee: Val::Func(d.coop_elem_ptr),
        args: ep_args,
        returns_value: true,
        fast: false,
    });

    let two_r = i32c(&mut b, 2);
    b.push(Inst::Call {
        fnty: t.t_run.clone(),
        callee: Val::Func(d.run),
        args: vec![
            desc_i8,
            sa.clone(),
            two_r.clone(),
            sb.clone(),
            two_r.clone(),
            dst_ptr,
            sg.clone(),
        ],
        returns_value: false,
        fast: false,
    });

    let mut store_args = desc_args(&mut b);
    store_args.extend([coop, sc.clone(), two_r, tag_c.clone(), tag_c, sg]);
    b.push(Inst::Call {
        fnty: t.t_store.clone(),
        callee: Val::Func(d.store),
        args: store_args,
        returns_value: false,
        fast: false,
    });

    b.push(Inst::Ret(None));
    b
}
