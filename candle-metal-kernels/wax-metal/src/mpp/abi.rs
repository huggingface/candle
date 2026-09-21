//! MPP ABI. Intrinsics, signatures, types, arguments, and conventions.
//!
//! Some conventions:
//! * Tensor descriptors and pointers are `i8*`. Resolved by name at pipeline creation.
//!   No type checking.
//! * `metal::array` is simply `[2 x i32]`. Already supported.
//! * Descriptors are built by via `{i32,i32,i32,i8,i8,i8,i32}`, five i32 words in an alloca,
//!   at 4 byte alignment.
//! * `alloca`s count operand is i32. Runtime descriptor sizes are zext/trunc to i32.

use super::config::{MppDtype, MppGemm};
use crate::bitcode::Ty;
use crate::bitcode::metadata::{Access, KernelArg};
use crate::lower::to_bitcode::Callee;
use crate::profile::TargetProfile;

/// MPP intrinsics
pub(super) const DESC_SIZE: &str = "air.get_descriptor_size_tensor";
pub(super) const SIMD_SIZE: &str = "air.get_simdgroup_size.i32";
pub(super) const INIT_STRIDED: &str = "air.init_strided_private_tensor.i32.global";
pub(super) const EXTENT: &str = "air.get_extent_private_tensor.i32";
pub(super) const SLICE: &str = "air.slice_private_tensor_private_tensor.s.i32";
pub(super) const SMIN: &str = "llvm.smin.i32";
pub(super) const COOP_SIZE: &str = "__tensorops_impl_matmul2d_op_cooperative_destination_data_size";
pub(super) const COOP_INIT: &str =
    "__tensorops_impl_matmul2d_op_cooperative_destination_tensor_init";
pub(super) const COOP_EP: &str =
    "__tensorops_impl_matmul2d_op_cooperative_destination_tensor_get_element_pointer";

pub(super) const EXTERN_DEF_SECTION: &str = "air.externally_defined";

pub(super) fn run(dtype: MppDtype, out: MppDtype) -> String {
    format!(
        "__tensorops_impl_matmul2d_op_run_cooperative_dv_{sfx}_dv_{sfx}_{osfx}",
        sfx = dtype.suffix(),
        osfx = out.suffix()
    )
}

pub(super) fn store(out: MppDtype) -> String {
    format!(
        "__tensorops_impl_matmul2d_op_cooperative_destination_tensor_store_dv_{osfx}",
        osfx = out.suffix()
    )
}

/// The declared ABI for the specific `MppGemm` config provided.
///
/// `air.externally_defined` function prototypes, entry function signature, pinned types,
/// and `air.kernel` arguments.
pub struct MppAbi {
    pub callees: Vec<Callee>,
    pub entry_params: Vec<Ty>,
    pub pinned_types: Vec<Ty>,
    pub args: Vec<KernelArg>,
}

// TODO: Find a use for `TargetProfile` or drop it.
pub fn mpp_abi(cfg: &MppGemm, _profile: &TargetProfile) -> MppAbi {
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

    /// This exists mostly for readability
    macro_rules! tys {
        ($($t:ident),* $(,)?) => { vec![$($t.clone()),*] };
    }

    // Tensor descriptor ABI, as defined in the Metal's tensor.metallib.
    let t_desc_size = (i16t.clone(), tys![i16t, i16t]);
    let t_simd_size = (i32t.clone(), vec![]);
    let t_init = (Ty::Void, tys![p0i8, i16t, p1i8, p0i8, p0i8, i8t]);
    let t_extent = (i32t.clone(), tys![p0i8, i16t, i16t]);
    let t_slice = (Ty::Void, tys![p0i8, p0i8, i16t, p0i8, p0i8]);
    let t_smin = (i32t.clone(), tys![i32t, i32t]);

    // The `air.externally_defined` matmul ABI.
    let dfields = || tys![i32t, i32t, i32t, i8t, i8t, i8t, i32t];
    let mut coop_size_p = dfields();
    coop_size_p.extend(tys![i32t, i32t, i32t, i32t]);
    let mut coop_init_p = dfields();
    coop_init_p.extend(tys![p0i8, i32t, i32t, i32t, i32t]);
    let mut coop_ep_p = dfields();
    coop_ep_p.extend(tys![p0i8, i16t, i32t, i32t, i32t]);
    let run_p = tys![p0i8, p0i8, i32t, p0i8, i32t, p0i8, i32t];
    let mut store_p = dfields();
    store_p.extend(tys![p0i8, p0i8, i32t, i32t, i32t, i32t]);

    let t_coop_size = (i64t.clone(), coop_size_p);
    let t_coop_init = (Ty::Void, coop_init_p);
    let t_coop_ep = (p0i8.clone(), coop_ep_p);
    let t_run = (Ty::Void, run_p);
    let t_store = (Ty::Void, store_p);

    // Defining the function prototypes
    let mut callees: Vec<Callee> = Vec::new();
    let mut add = |name: &str, (ret, params): (Ty, Vec<Ty>), section: Option<&'static str>| {
        let mut c = Callee::new(name, ret, params);
        if let Some(s) = section {
            c = c.in_section(s);
        }
        callees.push(c);
    };
    add(DESC_SIZE, t_desc_size, None);
    add(SIMD_SIZE, t_simd_size, None);
    add(INIT_STRIDED, t_init, None);
    add(EXTENT, t_extent, None);
    add(SLICE, t_slice, None);
    add(SMIN, t_smin, None);
    add(COOP_SIZE, t_coop_size, Some(EXTERN_DEF_SECTION));
    add(COOP_INIT, t_coop_init, Some(EXTERN_DEF_SECTION));
    add(COOP_EP, t_coop_ep, Some(EXTERN_DEF_SECTION));
    add(
        &run(cfg.dtype, cfg.out_dtype),
        t_run,
        Some(EXTERN_DEF_SECTION),
    );
    add(&store(cfg.out_dtype), t_store, Some(EXTERN_DEF_SECTION));

    let elem_size = cfg.dtype.size();
    let out_elem_size = cfg.out_dtype.size();
    let args = vec![
        KernelArg::Buffer {
            index: 0,
            address_space: 1,
            access: Access::ReadWrite,
            elem_size: out_elem_size,
            elem_align: out_elem_size,
            type_name: cfg.out_dtype.ll().to_string(),
            name: "c".into(),
        },
        KernelArg::Buffer {
            index: 1,
            address_space: 1,
            access: Access::Read,
            elem_size,
            elem_align: elem_size,
            type_name: cfg.dtype.ll().to_string(),
            name: "a".into(),
        },
        KernelArg::Buffer {
            index: 2,
            address_space: 1,
            access: Access::Read,
            elem_size,
            elem_align: elem_size,
            type_name: cfg.dtype.ll().to_string(),
            name: "b".into(),
        },
        KernelArg::Builtin {
            kind: "air.threadgroup_position_in_grid".into(),
            type_name: "uint3".into(),
            name: "tg".into(),
        },
    ];

    MppAbi {
        callees,
        entry_params: vec![p1oe, p1e.clone(), p1e, v3i32],
        // Pinned to match Metal's type table, not because it's functionally needed.
        pinned_types: vec![i64t, arr2],
        args,
    }
}
