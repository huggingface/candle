//! MPP as an AIR function.

use super::abi;
use super::config::{MppDtype, MppGemm};
use crate::air_dialect::attrs::{BinKind, CastOpAttr};
use crate::air_dialect::ops as air;
use crate::air_dialect::types as t;
use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::FunctionType;
use pliron::context::{Context, Ptr};
use pliron::irbuild::inserter::{IRInserter, Inserter};
use pliron::irbuild::listener::DummyListener;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::value::Value;
use std::collections::HashMap;

/// Emission state for [`mpp_air`].
struct MppAir<'a> {
    ctx: &'a mut Context,
    ins: IRInserter<DummyListener>,
    cfg: &'a MppGemm,
    consts: HashMap<(u32, i64), Value>,
}

/// How many parameters the MPP entry takes.
/// Destination, A, B, and the threadgroup builtin.
pub(crate) const ENTRY_PARAMS: usize = 4;

/// The air type an MPP element dtype lowers to.
// TODO: Yet another type conversion.
fn elem_ty(ctx: &Context, d: MppDtype) -> TypeHandle {
    match d {
        MppDtype::BF16 => t::bf16_ty(ctx),
        MppDtype::F16 => t::f16_ty(ctx),
        MppDtype::F32 => t::f32_ty(ctx),
    }
}

impl<'a> MppAir<'a> {
    /// The pliron i32 typehandle used to index and offset with.
    fn i32_t(&self) -> TypeHandle {
        t::i32_ty(self.ctx)
    }

    fn ptr(&self, pointee: TypeHandle, addrspace: u32) -> TypeHandle {
        t::ptr_ty(self.ctx, pointee, addrspace)
    }

    /// Output element type of this kernel.
    fn oelem(&self) -> TypeHandle {
        elem_ty(self.ctx, self.cfg.out_dtype)
    }

    /// Append op and return the first result.
    fn emit(&mut self, op: Ptr<Operation>) -> Value {
        self.ins.append_operation(self.ctx, op);
        op.deref(self.ctx).get_result(0)
    }

    fn emit_op(&mut self, o: impl Op) -> Value {
        self.emit(o.get_operation())
    }

    fn put(&mut self, o: impl Op) {
        self.ins.append_operation(self.ctx, o.get_operation());
    }

    fn const_i(&mut self, width: u32, v: i32) -> Value {
        let k = (width, v as i64);
        if let Some(x) = self.consts.get(&k) {
            return *x;
        }
        use wax_ir::dialect::ops::{BITS_8, BITS_16, BITS_32};
        let w = match width {
            8 => BITS_8,
            16 => BITS_16,
            _ => BITS_32,
        };
        let c = air::const_int(self.ctx, w, (v as i64) as u64);
        let r = self.emit_op(c);
        self.consts.insert(k, r);
        r
    }

    fn call(&mut self, name: &str, ret: TypeHandle, args: Vec<Value>) -> Value {
        let op = air::call(self.ctx, name, Some(ret), args);
        self.ins.append_operation(self.ctx, op);
        op.deref(self.ctx).get_result(0)
    }

    fn call_void(&mut self, name: &str, args: Vec<Value>) {
        let op = air::call(self.ctx, name, None, args);
        self.ins.append_operation(self.ctx, op);
    }

    fn cast(&mut self, to: TypeHandle, v: Value, op: CastOpAttr) -> Value {
        let c = air::cast(self.ctx, op, to, v);
        self.emit(c)
    }

    fn alloca_dyn(
        &mut self,
        ptr_h: TypeHandle,
        elem_h: TypeHandle,
        count: Value,
        align: u32,
    ) -> Value {
        let a = air::AllocaOp::new(self.ctx, ptr_h, Some(count), elem_h, None, align);
        self.emit_op(a)
    }

    fn alloca_static(
        &mut self,
        ptr_h: TypeHandle,
        elem_h: TypeHandle,
        count: u32,
        align: u32,
    ) -> Value {
        let a = air::AllocaOp::new(self.ctx, ptr_h, None, elem_h, Some(count), align);
        self.emit_op(a)
    }

    fn gep_arr(
        &mut self,
        res_h: TypeHandle,
        p: Value,
        idx: Value,
        elem_h: TypeHandle,
        count: u32,
    ) -> Value {
        let g = air::GepArrayOp::new(self.ctx, res_h, p, idx, elem_h, count);
        self.emit_op(g)
    }

    fn gep_el(&mut self, res_h: TypeHandle, p: Value, idx: Value, elem_h: TypeHandle) -> Value {
        let g = air::GepOp::new(self.ctx, res_h, p, idx, elem_h);
        self.emit_op(g)
    }

    fn store(&mut self, v: Value, p: Value, align: u32) {
        let s = air::StoreOp::new(self.ctx, v, p, align);
        self.put(s);
    }

    fn bin(&mut self, res_h: TypeHandle, a: Value, b: Value, op: BinKind) -> Value {
        let c = air::bin(self.ctx, op, res_h, a, b);
        self.emit(c)
    }

    fn extract(&mut self, res_h: TypeHandle, v: Value, idx: Value) -> Value {
        let c = air::extract_elem(self.ctx, res_h, v, idx);
        self.emit(c)
    }

    /// The matmul2d descriptor. BM, BN, BK, transA, transB, relaxed, mode.
    fn desc_args(&mut self) -> Vec<Value> {
        vec![
            self.const_i(32, self.cfg.bm),
            self.const_i(32, self.cfg.bn),
            self.const_i(32, -1),
            self.const_i(8, 0),
            self.const_i(8, i32::from(self.cfg.trans_b)),
            self.const_i(8, 1),
            self.const_i(32, 0),
        ]
    }

    /// Calculates split-K window and the plane of `c` it writes. `(origin, extent, base)`.
    /// `None` origin and extent means no split.
    fn split_window(&mut self, tg: Value, c: Value) -> (Option<Value>, Option<Value>, Value) {
        let (splits, k, m, n) = (self.cfg.splits.max(1), self.cfg.k, self.cfg.m, self.cfg.n);
        if splits <= 1 {
            return (None, None, c);
        }
        // `ceil(K/splits)`. Last split clamped to remainder.
        let ks = (k + splits - 1) / splits;
        let two_idx = self.const_i(32, 2);
        let s = self.extract(self.i32_t(), tg, two_idx);
        let ks_c = self.const_i(32, ks);
        let k0 = self.bin(self.i32_t(), s, ks_c, BinKind::Mul);
        let k_c = self.const_i(32, k);
        let krem = self.bin(self.i32_t(), k_c, k0, BinKind::Sub);
        let kext = self.call(abi::SMIN, self.i32_t(), vec![ks_c, krem]);
        let mn = self.const_i(32, m * n);
        let plane = self.bin(self.i32_t(), s, mn, BinKind::Mul);
        let cb = self.gep_el(self.ptr(self.oelem(), 1), c, plane, self.oelem());
        (Some(k0), Some(kext), cb)
    }

    /// Tile origin as `(m_off, n_off)`. Grid x indexes N, y indexes M.
    fn tile_origin(&mut self, tg: Value) -> (Value, Value) {
        let zero_idx = self.const_i(32, 0);
        let one_idx = self.const_i(32, 1);
        let tgy = self.extract(self.i32_t(), tg, one_idx);
        let bm_c = self.const_i(32, self.cfg.bm);
        let m_off = self.bin(self.i32_t(), tgy, bm_c, BinKind::Mul);
        let tgx = self.extract(self.i32_t(), tg, zero_idx);
        let bn_c = self.const_i(32, self.cfg.bn);
        let n_off = self.bin(self.i32_t(), tgx, bn_c, BinKind::Mul);
        (m_off, n_off)
    }

    /// 5 word matmul2d descriptor. Returned as `i8*`.
    fn descriptor_words(&mut self) -> Value {
        let desc = self.alloca_static(self.ptr(self.i32_t(), 0), self.i32_t(), 5, 4);
        let words = [
            self.cfg.bm,
            self.cfg.bn,
            -1,
            0x0001_0000 | (i32::from(self.cfg.trans_b) << 8),
            0,
        ];
        for (i, wv) in words.iter().enumerate() {
            // Leading zero is here to match Metal output.
            self.const_i(32, 0);
            let idx = self.const_i(32, i as i32);
            let g = self.gep_arr(self.ptr(self.i32_t(), 0), desc, idx, self.i32_t(), 5);
            let v = self.const_i(32, *wv);
            self.store(v, g, 4);
        }
        self.cast(self.ptr(t::i8_ty(self.ctx), 0), desc, CastOpAttr::Bitcast)
    }

    /// Static `[2 x i32]` pair, returned as `i8*`.
    fn mk_pair(&mut self, d0: Value, d1: Value) -> Value {
        let one = self.const_i(32, 1);
        let zero = self.const_i(32, 0);
        let p = self.alloca_static(self.ptr(self.i32_t(), 0), self.i32_t(), 2, 4);
        let g0 = self.gep_arr(self.ptr(self.i32_t(), 0), p, zero, self.i32_t(), 2);
        self.store(d0, g0, 4);
        let g1 = self.gep_arr(self.ptr(self.i32_t(), 0), p, one, self.i32_t(), 2);
        self.store(d1, g1, 4);
        self.cast(self.ptr(t::i8_ty(self.ctx), 0), p, CastOpAttr::Bitcast)
    }
}

/// MPP kernel for the provided [`MppGemm`] config as an `air` function.
pub fn mpp_air(cfg: &MppGemm, ctx: &mut Context) -> FuncOp {
    // Common types used below
    let i8_t = t::i8_ty(ctx);
    let i16_t = t::i16_ty(ctx);
    let i32_t = t::i32_ty(ctx);
    let i64_t = t::i64_ty(ctx);

    // The kernel function signature
    let in_ptr_t = t::ptr_ty(ctx, elem_ty(ctx, cfg.dtype), 1);
    let out_ptr_t = t::ptr_ty(ctx, elem_ty(ctx, cfg.out_dtype), 1);
    let vec3_i32 = t::vec_ty(ctx, 3, i32_t);
    let fn_ty = FunctionType::get(ctx, vec![out_ptr_t, in_ptr_t, in_ptr_t, vec3_i32], vec![]);

    // Create function op and extract the block args.
    let func = FuncOp::new(ctx, pliron::ident!("mpp"), fn_ty);
    let block = func.get_entry_block(ctx);
    let c_arg = block.deref(ctx).get_argument(0);
    let a_arg = block.deref(ctx).get_argument(1);
    let b_arg = block.deref(ctx).get_argument(2);
    let tg_arg = block.deref(ctx).get_argument(3);

    let mut e = MppAir {
        ctx,
        ins: IRInserter::new_at_block_end(block),
        cfg,
        consts: HashMap::new(),
    };

    // Common consts used below.
    let i8_2 = e.const_i(8, 2);
    let i16_0 = e.const_i(16, 0);
    let i16_1 = e.const_i(16, 1);
    let i16_2 = e.const_i(16, 2);
    let i16_4 = e.const_i(16, 4);
    let i32_neg1 = e.const_i(16, -1);
    let i32_0 = e.const_i(32, 0);
    let i32_1 = e.const_i(32, 1);
    let i32_2 = e.const_i(32, 2);

    let tag = cfg.dtype.tag();
    let out_tag = cfg.out_dtype.tag();

    // Descriptor sizes. Queried at runtime.
    let desc_size_i16 = e.call(abi::DESC_SIZE, i16_t, vec![i16_2, i16_4]);
    let desc_size = e.cast(i32_t, desc_size_i16, CastOpAttr::ZExt);
    let sg = e.call(abi::SIMD_SIZE, i32_t, vec![]);

    let mut coop_desc_args = e.desc_args();
    let const_tag = e.const_i(32, tag);
    let const_out_tag = e.const_i(32, out_tag);
    coop_desc_args.extend([const_tag, const_tag, const_out_tag, sg]);
    let coop_size_i64 = e.call(abi::COOP_SIZE, i64_t, coop_desc_args);
    let coop_size = e.cast(i32_t, coop_size_i64, CastOpAttr::Trunc);

    // Six tensor descriptors. 3 whole, 3 sliced, plus the cooperative destination.
    let mut tensors = Vec::new();
    for _ in 0..6 {
        tensors.push(e.alloca_dyn(e.ptr(i8_t, 0), i8_t, desc_size, 8));
    }
    let (ta, tb, tc) = (tensors[0], tensors[1], tensors[2]);
    let (sa, sb, sc) = (tensors[3], tensors[4], tensors[5]);
    let coop = e.alloca_dyn(e.ptr(i8_t, 0), i8_t, coop_size, 8);

    // Tensor extents are (columns, rows), dim 0 contiguous, with strides.
    let (k, m, n) = (cfg.k, cfg.m, cfg.n);
    let d_k = e.const_i(32, k);
    let d_m = e.const_i(32, m);
    let d_n = e.const_i(32, n);
    let d_lda = e.const_i(32, cfg.lda);
    let d_ldb = e.const_i(32, cfg.ldb);
    let d_ldc = e.const_i(32, cfg.ldc);
    let ea = e.mk_pair(d_k, d_m);
    let sla = e.mk_pair(i32_1, d_lda);
    let eb = if cfg.trans_b {
        e.mk_pair(d_k, d_n)
    } else {
        e.mk_pair(d_n, d_k)
    };
    let slb = e.mk_pair(i32_1, d_ldb);
    let ec = e.mk_pair(d_n, d_m);
    let slc = e.mk_pair(i32_1, d_ldc);

    let (k_origin, k_extent, cbase) = e.split_window(tg_arg, c_arg);

    // Initialise the three full tensor descriptors.
    for (tp, buf, ev, sv) in [
        (ta, a_arg, ea, sla),
        (tb, b_arg, eb, slb),
        (tc, cbase, ec, slc),
    ] {
        let i8p = e.cast(e.ptr(i8_t, 1), buf, CastOpAttr::Bitcast);
        e.call_void(abi::INIT_STRIDED, vec![tp, i16_2, i8p, ev, sv, i8_2]);
    }

    let (m_off, n_off) = e.tile_origin(tg_arg);

    // Slices. Extents run from origin to end of tensor. Op clamps to its own BM/BN, so edge tiles
    // need no special case.
    let k_o = k_origin.unwrap_or(i32_0);
    let (b_o0, b_o1, b_s0, b_s1, b_kdim) = if cfg.trans_b {
        (k_o, n_off, None, Some(n_off), 0usize)
    } else {
        (n_off, k_o, Some(n_off), None, 1)
    };
    for (dst, src, o0, o1, shrink0, shrink1, split_dim) in [
        (sa, ta, k_o, m_off, None, Some(m_off), 0usize),
        (sb, tb, b_o0, b_o1, b_s0, b_s1, b_kdim),
        (sc, tc, n_off, m_off, Some(n_off), Some(m_off), 2),
    ] {
        let origin = e.mk_pair(o0, o1);
        let x0 = e.call(abi::EXTENT, i32_t, vec![src, i16_2, i16_0]);
        let x1 = e.call(abi::EXTENT, i32_t, vec![src, i16_2, i16_1]);
        let e0 = if let Some(off) = shrink0 {
            e.bin(i32_t, x0, off, BinKind::Sub)
        } else if let Some(kext) = k_extent.filter(|_| split_dim == 0) {
            kext
        } else {
            x0
        };
        let e1 = if let Some(off) = shrink1 {
            e.bin(i32_t, x1, off, BinKind::Sub)
        } else if let Some(kext) = k_extent.filter(|_| split_dim == 1) {
            kext
        } else {
            x1
        };
        let extent = e.mk_pair(e0, e1);
        e.call_void(abi::SLICE, vec![dst, src, i16_2, origin, extent]);
    }

    // Cooperative destination tensor, the matmul, then the store.
    let mut init_args = e.desc_args();
    init_args.extend([coop, const_tag, const_tag, const_out_tag, sg]);
    // TODO: Would be a lot nicer if all these fns were self-describing.
    // struct CoopDestinationTensorInit;
    // impl ToAir for CoopDestinationTesorInit ...
    e.call_void(abi::COOP_INIT, init_args);

    let desc_i8 = e.descriptor_words();
    let mut ep_args = e.desc_args();
    ep_args.extend([coop, i32_neg1, const_tag, const_tag, const_out_tag]);
    let dst_ptr = e.call(abi::COOP_EP, e.ptr(i8_t, 0), ep_args);

    e.call_void(
        &abi::run(cfg.dtype, cfg.out_dtype),
        vec![desc_i8, sa, i32_2, sb, i32_2, dst_ptr, sg],
    );

    let mut store_args = e.desc_args();
    store_args.extend([coop, sc, i32_2, const_tag, const_tag, sg]);
    e.call_void(&abi::store(cfg.out_dtype), store_args);

    let ret = air::ReturnOp::new(e.ctx, None).get_operation();
    e.ins.append_operation(e.ctx, ret);
    func
}
