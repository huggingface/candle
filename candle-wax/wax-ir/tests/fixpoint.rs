use pliron::context::{Context, Ptr};
use pliron::irbuild::IRStatus;
use pliron::operation::Operation;
use pliron::pass::{AnalysisManager, Pass, PassResult};
use wax_frontend::KernelCompiler;
use wax_ir::passes::{Fixpoint, FixpointErr};

#[allow(unused_unsafe)]
#[wax_macro::module]
pub mod kernels {
    use cutile::core::*;

    #[wax_macro::entry()]
    pub fn scale<E: ElementType, const N: i32>(
        out: &mut Tensor<E, { [1, N] }>,
        x: &Tensor<E, { [1, N] }>,
    ) {
        let ts = const_shape![1, N];
        let px = x.partition(ts);
        let mut po: PartitionMut<E, { [1, N] }> = unsafe { out.partition_full_mut(ts) };
        let v = px.load([0i32, 0i32]);
        unsafe { po.store(v + v, [0i32, 0i32]) };
    }
}

fn scale_module() -> wax_ir::Module {
    KernelCompiler::new(kernels::__module_ast_self, "kernels", "scale")
        .generics(vec!["f32".into(), "32".into()])
        .strides(&[("out", &[32, 1]), ("x", &[32, 1])])
        .compile()
        .unwrap_or_else(|e| panic!("frontend: {e}"))
        .to_wax_module("scale")
        .unwrap_or_else(|e| panic!("wax module: {e}"))
}

struct ChangesTimes {
    left: usize,
}

impl Pass for ChangesTimes {
    fn name(&self) -> &str {
        "changes_times"
    }

    fn run(
        &mut self,
        _op: Ptr<Operation>,
        _ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        let mut r = PassResult::default();
        r.ir_changed = if self.left == 0 {
            IRStatus::Unchanged
        } else {
            self.left -= 1;
            IRStatus::Changed
        };
        Ok(r)
    }
}

fn run_to_fixpoint(
    changes: usize,
    max_iters: usize,
) -> (pliron::result::Result<PassResult>, usize) {
    let mut m = scale_module();
    let func = *m.entries().first().expect("must have entry function");
    let mut analyses = AnalysisManager::default();
    let mut fp = Fixpoint::new(ChangesTimes { left: changes }, max_iters);
    let res = fp.run(func, &mut m.ctx, &mut analyses);
    (res, fp.iters)
}

#[test]
fn pass_reports_changed_ir_correctly() {
    let (res, iters) = run_to_fixpoint(2, 8);
    let r = res.expect("pass reaches fixpoint");
    assert_eq!(r.ir_changed, IRStatus::Changed);
    assert_eq!(iters, 3);
}

#[test]
fn pass_reports_unchanged_ir_correctly() {
    let (res, iters) = run_to_fixpoint(0, 8);
    let r = res.expect("pass reaches fixpoint");
    assert_eq!(r.ir_changed, IRStatus::Unchanged);
    assert_eq!(iters, 1);
}

#[test]
fn pass_that_does_not_reach_fixpoint_is_error() {
    let (res, iters) = run_to_fixpoint(usize::MAX, 4);
    let Err(e) = res else {
        panic!("pass must fail");
    };
    assert!(
        e.err.downcast_ref::<FixpointErr>().is_some(),
        "incorrect error: {e}"
    );
    assert_eq!(iters, 4);
}
