//! Lowering `cross_loop_reuse` with thread-private cache.

use super::*;

/// Thread-private register budget for staging. Bytes per thread.
///
/// The IR pass records how many bytes is required for staging. Backend decides if the register
/// file can hold them.
const PRIV_BUDGET_BYTES: i64 = 128;

/// Cross-loop reuse staging plan.
/// Staging loop stores loaded results into a thread-private `[nc x elem]` cache.
/// Consuming loop reads from cache instead of repeating loads.
pub(crate) struct CachePlan {
    staging_for: Ptr<Operation>,
    consuming_for: Ptr<Operation>,
    nc: u32,
    elem: TypeHandle,
    elem_h: TypeHandle,
    align: u32,
    /// `air.alloca` created at the position of the staging loop.
    alloca: Option<pliron::value::Value>,
    pub(crate) staging_iv: Option<pliron::value::Value>,
    pub(crate) consuming_iv: Option<pliron::value::Value>,
}

impl CachePlan {
    /// The `elem addrspace(0)*` the cache alloca and its slot geps carry.
    fn ptr_ty(&self, ctx: &mut Context) -> TypeHandle {
        crate::air_dialect::types::ptr_ty(ctx, self.elem, 0)
    }
}

impl WaxToAir {
    /// Collect cross-loop reuse attributes from IR and store as plans.
    pub(crate) fn collect_cache_plans(&mut self, ctx: &mut Context, func: Ptr<Operation>) {
        use pliron::r#type::Typed;
        use wax_ir::types::Type as WT;
        struct RawPlan {
            staging_for: Ptr<Operation>,
            consuming_for: Ptr<Operation>,
            nc: i64,
            elem: TypeHandle,
            staging_loads: Vec<Ptr<Operation>>,
            consuming_loads: Vec<Ptr<Operation>>,
        }
        let mut raw: Vec<RawPlan> = Vec::new();
        for_each_block(ctx, func, &mut |ops| {
            let fors: Vec<Ptr<Operation>> = ops
                .iter()
                .copied()
                .filter(|&o| try_opcode_of(ctx, o) == Some(Opcode::For))
                .collect();
            for (idx, &staging) in fors.iter().enumerate() {
                let Some(nc) =
                    self.wax_attr_i64(ctx, staging, &crate::wax_to_air::keys::ATTR_CACHE_PRODUCER)
                else {
                    continue;
                };
                if self
                    .wax_attr_i64(ctx, staging, &crate::wax_to_air::keys::ATTR_CACHE_BYTES)
                    .is_none_or(|b| b > PRIV_BUDGET_BYTES)
                {
                    continue;
                }
                let Some(id) =
                    self.wax_attr_i64(ctx, staging, &crate::wax_to_air::keys::ATTR_CACHE_ID)
                else {
                    continue;
                };
                let Some(&consuming) = fors[idx + 1..].iter().find(|&&o| {
                    self.wax_attr_i64(ctx, o, &crate::wax_to_air::keys::ATTR_CACHE_CONSUMER)
                        .is_some()
                        && self.wax_attr_i64(ctx, o, &crate::wax_to_air::keys::ATTR_CACHE_ID)
                            == Some(id)
                }) else {
                    continue;
                };
                let views_s = self.body_load_views(ctx, staging, &self.lanes.lane_runs);
                let views_c = self.body_load_views(ctx, consuming, &self.lanes.lane_runs);
                let Some((pv, staging_loads)) = views_s
                    .iter()
                    .find(|(v, _)| views_c.iter().any(|(q, _)| q == v))
                    .cloned()
                else {
                    continue;
                };
                let Some(consuming_loads) = views_c
                    .iter()
                    .find(|(v, _)| *v == pv)
                    .map(|(_, l)| l.clone())
                else {
                    continue;
                };
                let Some(WT::PartitionView(pv_ty)) = wax_type_of(ctx, pv.get_type(ctx)) else {
                    continue;
                };
                let Some(elem) = crate::types::mem_elem_ty(ctx, &pv_ty.tensor_view.element_type)
                else {
                    continue;
                };
                raw.push(RawPlan {
                    staging_for: staging,
                    consuming_for: consuming,
                    nc,
                    elem,
                    staging_loads,
                    consuming_loads,
                });
            }
        });
        for r in raw {
            let elem_h = r.elem;
            // Alignment required for stride which is required for cache.
            let Some(align) = crate::air_dialect::type_interfaces::align_bytes(ctx, r.elem) else {
                continue;
            };
            let pi = self.staging.cache_plans.len();
            self.staging.cache_plans.push(CachePlan {
                staging_for: r.staging_for,
                consuming_for: r.consuming_for,
                nc: r.nc as u32,
                elem: r.elem,
                elem_h,
                align,
                alloca: None,
                staging_iv: None,
                consuming_iv: None,
            });
            for &l in &r.staging_loads {
                self.staging.cache_members.insert(l, (pi, true));
            }
            for &l in &r.consuming_loads {
                self.staging.cache_members.insert(l, (pi, false));
            }
            self.staging.cache_fors.insert(r.staging_for, (pi, true));
            self.staging.cache_fors.insert(r.consuming_for, (pi, false));
        }
    }

    /// The (view, loads) pairs of a loop body.
    // TODO: investigate if simd-matrix loads and/or lane-vector members can be cached similarly.
    pub(crate) fn body_load_views(
        &self,
        ctx: &Context,
        for_op: Ptr<Operation>,
        lane_runs: &std::collections::HashMap<Ptr<Operation>, (usize, usize)>,
    ) -> Vec<(pliron::value::Value, Vec<Ptr<Operation>>)> {
        use pliron::linked_list::ContainsLinkedList;
        use pliron::r#type::Typed;
        let r = for_op.deref(ctx).get_region(0);
        let mut out: Vec<(pliron::value::Value, Vec<Ptr<Operation>>)> = Vec::new();
        for b in r.deref(ctx).iter(ctx) {
            for op in b.deref(ctx).iter(ctx) {
                if try_opcode_of(ctx, op) != Some(Opcode::LoadViewTko) {
                    continue;
                }
                let o = op.deref(ctx);
                let Some(view) = o.operands().next() else {
                    continue;
                };
                let Some(wt) = wax_type_of(ctx, view.get_type(ctx)) else {
                    continue;
                };
                let is_pv = matches!(wt, wax_ir::types::Type::PartitionView(_));
                let res0 = (o.get_num_results() > 0).then(|| o.get_result(0));
                let is_frag = res0
                    .and_then(|v| wax_type_of(ctx, v.get_type(ctx)))
                    .map(|t| crate::types::sgmma_dims(&t).is_some())
                    .unwrap_or(false);
                if !is_pv || is_frag || lane_runs.contains_key(&op) {
                    continue;
                }
                match out.iter_mut().find(|(q, _)| *q == view) {
                    Some((_, l)) => l.push(op),
                    None => out.push((view, vec![op])),
                }
            }
        }
        out
    }

    /// The induction variable (IV) a cache slot indexes with.
    /// Uses the loop body's IV until the `replace_all_uses_with` rewrite substitutes it with
    /// the header argument. This means  slot gep built before the loop exists points to the
    /// correct value after.
    pub(crate) fn cache_iv(
        &self,
        ctx: &Context,
        pi: usize,
        is_staging: bool,
    ) -> Option<pliron::value::Value> {
        use pliron::linked_list::ContainsLinkedList;
        let p = &self.staging.cache_plans[pi];
        let ready = if is_staging {
            p.staging_iv
        } else {
            p.consuming_iv
        };
        if ready.is_some() {
            return ready;
        }
        let for_op = if is_staging {
            p.staging_for
        } else {
            p.consuming_for
        };
        let r = for_op.deref(ctx).get_region(0);
        let mut it = r.deref(ctx).iter(ctx);
        let b = it.next()?;
        it.next().is_none().then(|| b.deref(ctx).get_argument(0))
    }

    /// Ensure the thread-private cache exists at the staging loop's position.
    pub(crate) fn ensure_cache_alloca(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        pi: usize,
    ) -> Result<()> {
        use pliron::irbuild::inserter::OpInsertionPoint;
        if self.staging.cache_plans[pi].alloca.is_some() {
            return Ok(());
        }
        let (pos, align, nc, elem_h) = {
            let p = &self.staging.cache_plans[pi];
            (p.staging_for, p.align, p.nc, p.elem_h)
        };
        rewriter.set_insertion_point(OpInsertionPoint::BeforeOperation(pos));
        let ptr_h = self.staging.cache_plans[pi].ptr_ty(ctx);
        let a = air::AllocaOp::new(ctx, ptr_h, None, elem_h, Some(nc), align).get_operation();
        rewriter.insert_operation(ctx, a);
        self.staging.cache_plans[pi].alloca = Some(a.deref(ctx).get_result(0));
        Ok(())
    }

    /// Get the cache address for slot `iv`.
    ///
    /// Creates the cache first if it does not exist.
    pub(crate) fn cache_addr(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        pi: usize,
        iv: pliron::value::Value,
    ) -> Result<Option<pliron::value::Value>> {
        if self.staging.cache_plans[pi].alloca.is_none() {
            let saved = rewriter.get_insertion_point();
            self.ensure_cache_alloca(ctx, rewriter, pi)?;
            rewriter.set_insertion_point(saved);
        }
        self.cache_gep(ctx, rewriter, pi, iv)
    }

    /// `air.gep_array` into the cache at slot `iv`.
    pub(crate) fn cache_gep(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        pi: usize,
        iv: pliron::value::Value,
    ) -> Result<Option<pliron::value::Value>> {
        let (alloca, ptr_h, elem_h, nc) = {
            let p = &self.staging.cache_plans[pi];
            let Some(alloca) = p.alloca else {
                return Ok(None);
            };
            (alloca, p.ptr_ty(ctx), p.elem_h, p.nc)
        };
        let g = air::GepArrayOp::new(ctx, ptr_h, alloca, iv, elem_h, nc);
        rewriter.insert_operation(ctx, g.get_operation());
        Ok(Some(g.get_operation().deref(ctx).get_result(0)))
    }

    /// Rewrite consumer loop member to use cached load.
    pub(crate) fn rewrite_cached_load(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        pi: usize,
        res_tys: &[TypeHandle],
    ) -> Result<()> {
        let Some(iv) = self.cache_iv(ctx, pi, false) else {
            self.bail(Opcode::LoadViewTko, "cache loop not rewritten");
            return Ok(());
        };
        let Some(addr) = self.cache_addr(ctx, rewriter, pi, iv)? else {
            self.bail(Opcode::LoadViewTko, "cache alloca missing");
            return Ok(());
        };
        let Some(&res_ty) = res_tys.first() else {
            self.bail(Opcode::LoadViewTko, "load-result-missing");
            return Ok(());
        };
        let Some(res_ty) = self.air_ty(ctx, res_ty) else {
            self.bail(Opcode::LoadViewTko, "load-result-type");
            return Ok(());
        };
        let align = self.staging.cache_plans[pi].align;
        let ld = air::LoadOp::new(ctx, res_ty, addr, align);
        rewriter.insert_operation(ctx, ld.get_operation());
        let val = ld.get_operation().deref(ctx).get_result(0);
        self.finish_replacement(ctx, rewriter, op, res_tys, vec![val]);
        Ok(())
    }

    /// Emit cache store after staging loop member load.
    pub(crate) fn emit_cache_store(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        pi: usize,
        val: pliron::value::Value,
    ) -> Result<()> {
        let Some(iv) = self.cache_iv(ctx, pi, true) else {
            self.bail(Opcode::LoadViewTko, "cache loop not rewritten");
            return Ok(());
        };
        let Some(addr) = self.cache_addr(ctx, rewriter, pi, iv)? else {
            self.bail(Opcode::LoadViewTko, "cache alloca missing");
            return Ok(());
        };
        let align = self.staging.cache_plans[pi].align;
        let st = air::StoreOp::new(ctx, val, addr, align);
        rewriter.insert_operation(ctx, st.get_operation());
        Ok(())
    }
}
