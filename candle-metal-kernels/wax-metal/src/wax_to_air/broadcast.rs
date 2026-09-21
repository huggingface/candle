//! `wax.broadcast`: forward when a lane already holds its element, else redistribute.

use super::*;
use crate::air_dialect::attrs::{BinKind, CastOpAttr};
use crate::air_dialect::types as t;
use pliron::r#type::Typed;
use pliron::value::Value;

/// The scratch is an f32 array. A 0 here encodes as `align 4294967296`, not as "unspecified".
const SCRATCH_ALIGN: u32 = 4;

impl WaxToAir {
    pub(crate) fn rewrite_broadcast(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let base = op.deref(ctx).operands().next();
        let shapes = (|| {
            let base = base?;
            let dst_handle = {
                let o = op.deref(ctx);
                (o.get_num_results() > 0)
                    .then(|| o.get_result(0))?
                    .get_type(ctx)
            };
            let WaxType::Tile(src_tile) = self.wax_operand_ty(ctx, operands_info, base)? else {
                return None;
            };
            let WaxType::Tile(dst_tile) = wax_type_of(ctx, dst_handle)? else {
                return None;
            };
            Some((src_tile.shape, dst_tile.shape))
        })();
        let Some((src, dst)) = shapes else {
            self.bail(
                Opcode::Broadcast,
                "broadcast operand or result is not a tile",
            );
            return Ok(());
        };
        let replicated = base.is_some_and(|v| self.reduce.row_replicated.contains(&v));
        if broadcast_is_forwardable(&src, &dst)
            || (replicated && expands_replicated_row(&src, &dst))
        {
            return self.fold_view(ctx, rewriter, op, Opcode::Broadcast);
        }
        self.redistribute_broadcast(ctx, rewriter, op, &src, &dst)
    }

    fn redistribute_broadcast(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        src: &[i64],
        dst: &[i64],
    ) -> Result<()> {
        let src = right_align(src, dst);
        let Some(terms) = broadcast_terms(&src, dst) else {
            self.bail(Opcode::Broadcast, "shapes are not in a broadcast relation");
            return Ok(());
        };
        let Some(value) = op.deref(ctx).operands().next() else {
            self.bail(Opcode::Broadcast, "no source operand");
            return Ok(());
        };
        let Some(tp) = self.control.tp else {
            self.bail(Opcode::Broadcast, "no thread position parameter");
            return Ok(());
        };
        let elem_h = value.get_type(ctx);
        let Some(casts) = scratch_casts(ctx, elem_h) else {
            self.bail(
                Opcode::Broadcast,
                "source element does not ride in the scratch",
            );
            return Ok(());
        };
        let (n_src, n_dst): (i64, i64) = (src.iter().product(), dst.iter().product());
        // `+1` for dump slot.
        if i64::from(self.reduce.tg_scratch_floats) < n_src + 1 {
            self.bail(Opcode::Broadcast, "source tile does not fit the scratch");
            return Ok(());
        }
        let Some(rt) = self.scratch_round_trip(ctx, rewriter)? else {
            return Ok(());
        };
        let i32_h = t::i32_ty(ctx);
        let zero = air_const_i32(ctx, 0);
        let e = air::extract_elem(ctx, i32_h, tp, zero);
        let tid = emit(ctx, rewriter, e);

        scratch_write(ctx, rewriter, &rt, tid, value, n_src, casts.0);
        air_wg_barrier(ctx, rewriter);
        let out = scratch_read(ctx, rewriter, &rt, tid, &terms, n_dst, elem_h, casts.1);

        rewriter.replace_operation_with_values(ctx, op, vec![out]);
        self.converted += 1;
        Ok(())
    }
}

fn scratch_write(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    rt: &ScratchRoundTrip,
    tid: Value,
    value: Value,
    n_src: i64,
    to_f32: Option<CastOpAttr>,
) {
    let (i32_h, i1_h, f32_h) = (t::i32_ty(ctx), t::i1_ty(ctx), t::f32_ty(ctx));
    let n_c = air_const_i32(ctx, n_src);
    // A lane past the source writes to dump slot rather than skipping, so store remains
    // unconditional and uniform.
    let c = air::icmp(
        ctx,
        pliron_llvm::attributes::ICmpPredicateAttr::ULT,
        i1_h,
        tid,
        n_c,
    );
    let keep = emit(ctx, rw, c);
    let s = air::select(ctx, i32_h, keep, tid, n_c);
    let store_idx = emit(ctx, rw, s);

    let word = maybe_cast(ctx, rw, to_f32, f32_h, value);
    let p = rt.scratch_gep(ctx, rw, store_idx);
    let st = air::StoreOp::new(ctx, word, p, SCRATCH_ALIGN);
    rw.insert_operation(ctx, st.get_operation());
}

#[allow(clippy::too_many_arguments)]
fn scratch_read(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    rt: &ScratchRoundTrip,
    tid: Value,
    terms: &[Term],
    n_dst: i64,
    elem_h: TypeHandle,
    from_f32: Option<CastOpAttr>,
) -> Value {
    let f32_h = t::f32_ty(ctx);
    let mut acc: Option<Value> = None;
    for &(dst_d, inner_d, inner_s) in terms {
        let mut c = tid;
        if inner_d > 1 {
            let k = air_const_i32(ctx, inner_d);
            c = air_bin_i32(ctx, rw, c, k, BinKind::UDiv);
        }
        // Redundant once nothing outer is left to strip.
        if inner_d * dst_d != n_dst {
            let k = air_const_i32(ctx, dst_d);
            c = air_bin_i32(ctx, rw, c, k, BinKind::URem);
        }
        if inner_s > 1 {
            let k = air_const_i32(ctx, inner_s);
            c = air_bin_i32(ctx, rw, c, k, BinKind::Mul);
        }
        acc = Some(match acc {
            None => c,
            Some(a) => air_bin_i32(ctx, rw, a, c, BinKind::Add),
        });
    }
    let load_idx = acc.unwrap_or_else(|| air_const_i32(ctx, 0));
    let p = rt.scratch_gep(ctx, rw, load_idx);
    let ld = air::LoadOp::new(ctx, f32_h, p, SCRATCH_ALIGN);
    let got = emit_op(ctx, rw, ld);
    maybe_cast(ctx, rw, from_f32, elem_h, got)
}

/// How elements are stored in f32 scratch as `(in, out)`.
// TODO: Extend to store in other types than f32.
fn scratch_casts(
    ctx: &mut Context,
    elem: TypeHandle,
) -> Option<(Option<CastOpAttr>, Option<CastOpAttr>)> {
    if elem == t::f32_ty(ctx) {
        Some((None, None))
    } else if elem == t::i32_ty(ctx) {
        Some((Some(CastOpAttr::Bitcast), Some(CastOpAttr::Bitcast)))
    } else if elem == t::f16_ty(ctx) || elem == t::bf16_ty(ctx) {
        Some((Some(CastOpAttr::FPExt), Some(CastOpAttr::FPTrunc)))
    } else {
        None
    }
}

fn maybe_cast(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    k: Option<CastOpAttr>,
    ty: TypeHandle,
    v: Value,
) -> Value {
    match k {
        None => v,
        Some(k) => {
            let c = air::cast(ctx, k, ty, v);
            emit(ctx, rw, c)
        }
    }
}

fn right_align(src: &[i64], dst: &[i64]) -> Vec<i64> {
    std::iter::repeat_n(1, dst.len().saturating_sub(src.len()))
        .chain(src.iter().copied())
        .collect()
}

/// `(dst[d], inner_d, inner_s)`.
type Term = (i64, i64, i64);

/// The source index a destination index `flat` wants is
/// `sum over terms of ((flat / inner_d) % dst_d) * inner_s`.
/// Expanded dim yields no term. `src` must already be right-aligned.
fn broadcast_terms(src: &[i64], dst: &[i64]) -> Option<Vec<Term>> {
    // A zero extent would divide by zero below. `broadcast_is_forwardable` rejects one, which
    // routes it straight here.
    if src.len() != dst.len() || src.iter().chain(dst).any(|&e| e <= 0) {
        return None;
    }
    let (mut inner_d, mut inner_s): (i64, i64) = (dst.iter().product(), src.iter().product());
    let mut out = Vec::new();
    for d in 0..dst.len() {
        inner_d /= dst[d];
        inner_s /= src[d];
        if src[d] == dst[d] {
            out.push((dst[d], inner_d, inner_s));
        } else if src[d] != 1 {
            return None;
        }
    }
    Some(out)
}

/// Whether every lane already holds the element the destination wants, making the broadcast just
/// its operand.
/// True iff the expanded dims form a prefix. The kept suffix then shares its strides with the
/// destination, so lane `tp` wants element `tp % n_src`, which is what its lane term
/// `tp & (n_src - 1)` already computes when `n_src` is a power of two.
pub(crate) fn broadcast_is_forwardable(src: &[i64], dst: &[i64]) -> bool {
    let (src_n, dst_n): (i64, i64) = (src.iter().product(), dst.iter().product());
    if src_n <= 0 || dst_n <= 0 || src.len() > dst.len() {
        return false;
    }
    let a = right_align(src, dst);
    let mut kept = false;
    for d in 0..dst.len() {
        if dst[d] == 1 {
            if a[d] != 1 {
                return false;
            }
        } else if a[d] == dst[d] {
            kept = true;
        } else if a[d] != 1 || kept {
            return false;
        }
    }
    (src_n as u64).is_power_of_two()
}

/// Whether `dst` is `src` with its last dim opened from 1 to the 32 lanes a row-replicated
/// value already occupies.
fn expands_replicated_row(src: &[i64], dst: &[i64]) -> bool {
    dst.len() == 2
        && dst[1] == 32
        && match src.len() {
            1 => src[0] == dst[0],
            2 => src[0] == dst[0] && src[1] == 1,
            _ => false,
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src_flat(terms: &[Term], flat: i64) -> i64 {
        terms
            .iter()
            .map(|&(dst_d, inner_d, inner_s)| ((flat / inner_d) % dst_d) * inner_s)
            .sum()
    }

    fn forwardable_by_enumeration(src: &[i64], dst: &[i64]) -> bool {
        let (src_n, dst_n): (i64, i64) = (src.iter().product(), dst.iter().product());
        if src_n <= 0 || dst_n <= 0 || src.len() > dst.len() {
            return false;
        }
        let Some(terms) = broadcast_terms(&right_align(src, dst), dst) else {
            return false;
        };
        (0..dst_n).all(|f| src_flat(&terms, f) == f & (src_n - 1))
    }

    fn shapes(rank: usize) -> Vec<Vec<i64>> {
        const DIMS: [i64; 5] = [1, 2, 3, 4, 8];
        let mut out = vec![vec![]];
        for _ in 0..rank {
            out = out
                .iter()
                .flat_map(|p| DIMS.iter().map(|&d| [p.as_slice(), &[d]].concat()))
                .collect();
        }
        out
    }

    #[test]
    fn broadcast_is_forwardable_matches_enumeration_approach() {
        for rank in 1..=3 {
            for dst in shapes(rank) {
                if dst.iter().product::<i64>() > 256 {
                    continue;
                }
                for sr in 1..=rank {
                    for src in shapes(sr) {
                        assert_eq!(
                            broadcast_is_forwardable(&src, &dst),
                            forwardable_by_enumeration(&src, &dst),
                            "{src:?} -> {dst:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn broadcast_terms_are_calculated_correctly() {
        for (src, dst) in [
            (vec![1i64, 4], vec![3i64, 4]),
            (vec![3, 1], vec![3, 4]),
            (vec![4], vec![3, 4]),
            (vec![1, 1], vec![3, 4]),
            (vec![2, 3, 1], vec![2, 3, 5]),
        ] {
            let a = right_align(&src, &dst);
            let terms = broadcast_terms(&a, &dst).expect("a broadcast relation");
            let n_dst: i64 = dst.iter().product();
            for flat in 0..n_dst {
                let (mut rest, mut coords) = (flat, vec![0i64; dst.len()]);
                for d in (0..dst.len()).rev() {
                    coords[d] = rest % dst[d];
                    rest /= dst[d];
                }
                let (mut want, mut stride) = (0i64, 1i64);
                for d in (0..dst.len()).rev() {
                    if a[d] == dst[d] {
                        want += coords[d] * stride;
                    }
                    stride *= a[d];
                }
                assert_eq!(src_flat(&terms, flat), want, "{src:?} -> {dst:?} at {flat}");
            }
        }
    }

    #[test]
    fn broadcast_forwarding_is_rejected_correctly() {
        assert!(broadcast_terms(&right_align(&[2, 3], &[2, 4]), &[2, 4]).is_none());
        assert!(broadcast_terms(&right_align(&[2, 3, 4], &[3, 4]), &[3, 4]).is_none());
        assert!(!broadcast_is_forwardable(&[2, 3, 4], &[3, 4]));
    }

    #[test]
    fn broadcast_is_forwardable_when_all_lanes_are_uniform() {
        assert!(broadcast_is_forwardable(&[4], &[4]));
        assert!(!broadcast_is_forwardable(&[4, 1], &[4, 32]));
    }
}
