//! Run a pass (passes, graph of passes) until IR is unchanged.

use pliron::context::{Context, Ptr};
use pliron::irbuild::IRStatus;
use pliron::operation::Operation;
use pliron::pass::{AnalysisManager, Pass, PassManager, PassResult};
use pliron::result::Result;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FixpointErr {
    #[error("`{pass}` exceeded max iterations: {max_iters}")]
    NotReached { pass: String, max_iters: usize },
}

/// Repeats `inner` until it reports [`IRStatus::Unchanged`].
///
/// `inner` is any pass. Could be a single pass or an entire graph of passes composed by `Passes`,
/// `GuardedPass`, `NestedOpsPass`, or any other combination of passes. Being a `Pass` itself, a
/// `Fixpoint` nests in turn.
pub struct Fixpoint<P: Pass> {
    inner: P,
    max_iters: usize,
    /// Counted iterations during `run`, for tests/diagnostics.
    pub iters: usize,
}

impl<P: Pass> Fixpoint<P> {
    pub fn new(inner: P, max_iters: usize) -> Self {
        Fixpoint {
            inner,
            max_iters,
            iters: 0,
        }
    }
}

impl<P: Pass> Pass for Fixpoint<P> {
    fn name(&self) -> &str {
        "fixpoint"
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        // Run passes until IR is unchanged

        let mut changed = IRStatus::Unchanged;
        self.iters = 0;
        for _ in 0..self.max_iters {
            let mut res = <Self as PassManager>::run_pass(&mut self.inner, op, ctx, analyses)?;
            analyses.retain_preserved(&res);
            self.iters += 1;
            changed |= res.ir_changed;
            if res.ir_changed == IRStatus::Unchanged {
                res.ir_changed = changed;
                return Ok(res);
            }
        }
        pliron::verify_err_noloc!(FixpointErr::NotReached {
            pass: self.inner.name().to_string(),
            max_iters: self.max_iters,
        })
    }

    fn as_pass_manager(&mut self) -> Option<&mut dyn PassManager> {
        Some(self)
    }
}

impl<P: Pass> PassManager for Fixpoint<P> {}
