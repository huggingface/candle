//! LLVM bitcode writer scaffolding.

use crate::bitcode::func::Body;
use crate::bitcode::metadata::{KernelArg, KernelMetadata};
use crate::bitcode::{Const, Module, Ty};
use crate::profile::TargetProfile;

/// Why bitcode encoding failed
// TODO: Rename to something like `EncodingError`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Unsupported {
    #[error("unmapped opcode `{0}`")]
    Opcode(String),
    #[error("undeclared callee `{0}`")]
    UnknownCallee(String),
    #[error("unmapped type `{0}`")]
    Type(String),
    #[error("unresolved value at `{op}`{at}", at = .loc.as_deref().map(|l| format!(" ({l})")).unwrap_or_default())]
    UnresolvedValue { op: String, loc: Option<String> },
    #[error("`{op}` has no `{attr}`")]
    MissingAttr {
        op: &'static str,
        attr: &'static str,
    },
}

impl Unsupported {
    pub(crate) fn missing(op: &'static str, attr: &'static str) -> Self {
        Unsupported::MissingAttr { op, attr }
    }
}

/// Prototype function. LLVM `declare`.
// TODO: Rename to something like `Prototype`/`Declare`. Maybe mix `Func` in.
#[derive(Clone)]
pub struct Callee {
    pub name: String,
    pub ret: Ty,
    pub params: Vec<Ty>,
    /// Potential section reference needed by driver to supply section body at pipeline creation.
    /// MPP uses `"air.externally_defined"` for tensorops primitives.
    pub section: Option<&'static str>,
}

impl Callee {
    pub fn new(name: impl AsRef<str>, ret: Ty, params: Vec<Ty>) -> Self {
        Self {
            name: name.as_ref().to_string(),
            ret,
            params,
            section: None,
        }
    }
    pub fn in_section(mut self, section: &'static str) -> Self {
        self.section = Some(section);
        self
    }
    pub fn ty(&self) -> Ty {
        Ty::Func {
            ret: Box::new(self.ret.clone()),
            params: self.params.clone(),
            vararg: false,
        }
    }
}

/// Module under bitcode assembly
pub struct BitcodeModuleAssembler<'a> {
    callees: Vec<Callee>,
    globals: Vec<(String, Ty, u32)>,
    kernels: Vec<BuiltKernel<'a>>,
    profile: &'a TargetProfile,
}

impl<'a> BitcodeModuleAssembler<'a> {
    pub fn new(profile: &'a TargetProfile) -> Self {
        BitcodeModuleAssembler {
            callees: Vec::new(),
            globals: Vec::new(),
            kernels: Vec::new(),
            profile,
        }
    }
    /// Get or insert callee, returning index.
    pub fn declare_callee(&mut self, c: Callee) -> u32 {
        match self.callees.iter().position(|x| x.name == c.name) {
            Some(i) => i as u32,
            None => {
                self.callees.push(c);
                (self.callees.len() - 1) as u32
            }
        }
    }
    /// Get or insert global, returning index.
    pub fn declare_global(&mut self, name: String, ty: Ty, count: u32) -> u32 {
        match self.globals.iter().position(|g| g.0 == name) {
            Some(i) => i as u32,
            None => {
                self.globals.push((name, ty, count));
                (self.globals.len() - 1) as u32
            }
        }
    }
    /// Get callee index by name, if present.
    pub fn callee_idx(&self, name: &str) -> Option<u32> {
        self.callees
            .iter()
            .position(|c| c.name == name)
            .map(|i| i as u32)
    }
    /// Get global index by name, if present.
    pub fn global_idx(&self, name: &str) -> Option<u32> {
        self.globals
            .iter()
            .position(|g| g.0 == name)
            .map(|i| i as u32)
    }

    /// Add a kernel to the bitcode module.
    ///
    /// The default `Ty::Int(64)` pinned type is there simply to ensure our type table is ordered
    /// that same way Metal does.
    pub fn add_kernel(
        &mut self,
        name: &'a str,
        param_types: Vec<Ty>,
        args: &'a [KernelArg],
        body: Body,
    ) {
        self.add_kernel_pinning(name, param_types, args, body, vec![Ty::Int(64)]);
    }

    pub fn add_kernel_pinning(
        &mut self,
        name: &'a str,
        param_types: Vec<Ty>,
        args: &'a [KernelArg],
        body: Body,
        pinned_types: Vec<Ty>,
    ) {
        self.kernels.push(BuiltKernel {
            name,
            param_types,
            args,
            body,
            pinned_types,
        });
    }

    /// Assembled module, before serialisation.
    pub fn build(self) -> Result<Module, Unsupported> {
        build_bitcode_module(&self.callees, self.kernels, &self.globals, self.profile)
    }

    /// Build and emit as bitcode.
    pub fn finish(self) -> Result<Vec<u8>, Unsupported> {
        self.build()?.to_bitcode().map_err(uninterned)
    }
}

/// One built kernel of a module.
pub struct BuiltKernel<'a> {
    pub name: &'a str,
    pub param_types: Vec<Ty>,
    pub args: &'a [KernelArg],
    pub body: Body,
    /// Type table is ordered by usage. Pinning types allows us to ensure correct ordering.
    pub pinned_types: Vec<Ty>,
}

// TODO: Only used in tests. Remove.
pub fn bitcode_with_kernels(
    callees: &[Callee],
    kernels: Vec<BuiltKernel>,
    tg_globals: &[(String, Ty, u32)],
    profile: &TargetProfile,
) -> Result<Vec<u8>, Unsupported> {
    build_bitcode_module(callees, kernels, tg_globals, profile)?
        .to_bitcode()
        .map_err(uninterned)
}

/// Returns the built bitcode module, before serialization.
/// Globals, function prototypes, kernel bodies, and metadata.
pub fn build_bitcode_module(
    callees: &[Callee],
    kernels: Vec<BuiltKernel>,
    tg_globals: &[(String, Ty, u32)],
    profile: &TargetProfile,
) -> Result<Module, Unsupported> {
    let mut m = Module::new(profile.triple(), profile.datalayout());
    let ia = m.intrinsic_attrs();
    let ka = m.kernel_attrs();

    // Globals
    for (name, elem, count) in tg_globals {
        // Strip `@` prefix for symbol table.
        m.threadgroup_global(name.trim_start_matches('@'), elem.clone(), *count as u64);
    }
    // Function prototypes.
    for c in callees {
        let ty = c.ty();
        // Strip `@` prefix for symbol table.
        let name = c.name.trim_start_matches('@');
        match c.section {
            Some(sec) => m.declare_in_section(name, ty, true, ia, sec),
            None => m.declare(name, ty, true, ia),
        }
    }
    // Validate all kernel bodies.
    for k in &kernels {
        k.body
            .validate()
            .map_err(|e| Unsupported::Opcode(format!("{}: {e}", k.name)))?;
    }

    // Kernel entry functions.
    let mut slots = Vec::new();
    for k in &kernels {
        let kty = Ty::Func {
            ret: Box::new(Ty::Void),
            params: k.param_types.clone(),
            vararg: false,
        };
        let entry_slot = m.funcs.len();
        m.declare(k.name, kty, false, ka);
        for t in &k.pinned_types {
            m.types.intern(t);
        }
        slots.push(entry_slot);
    }
    let const_base = m.const_base();
    for (i, k) in kernels.into_iter().enumerate() {
        let entry_slot = slots[i];
        let kty = Ty::Func {
            ret: Box::new(Ty::Void),
            params: k.param_types.clone(),
            vararg: false,
        };
        let mut body = k.body;

        // `!llvm.loop` unroll hints.
        // TODO: extract?
        if !body.loop_hints.is_empty() {
            let kind = m.md_kinds.intern("llvm.loop");
            let enable = m.md.string("llvm.loop.unroll.enable");
            let enable_node = m.md.tuple(&[enable]);
            let count_key = m.md.string("llvm.loop.unroll.count");
            let hints = std::mem::take(&mut body.loop_hints);
            for (idx, count) in hints {
                m.types.intern(&Ty::Int(32));
                let cid = m.consts.intern(Const::Int(Ty::Int(32), count as i64));
                let cval = m.md.value(Ty::Int(32), const_base + cid);
                let count_node = m.md.tuple(&[count_key, cval]);
                let loop_id = m.md.distinct_node();
                m.md.set_node(loop_id, &[loop_id.into(), enable_node, count_node]);
                body.attach(idx, kind, loop_id.into());
            }
        }
        m.set_body(entry_slot, body);

        let km = KernelMetadata {
            entry_value_id: m.func_value_id(entry_slot as u32),
            entry_fn_ptr_ty: Ty::ptr(kty, 0),
            args: k.args,
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
    }

    // Module built. Validate before returning.
    m.check_types().map_err(uninterned)?;
    Ok(m)
}

fn uninterned(e: String) -> Unsupported {
    Unsupported::Type(format!("{e} (the writer would have panicked)"))
}

#[cfg(test)]
mod tests {
    use super::{BuiltKernel, Callee, build_bitcode_module};
    use crate::bitcode::Ty;
    use crate::bitcode::func::{Body, Inst};
    use crate::profile::{Platform, TargetProfile};

    fn profile() -> TargetProfile {
        TargetProfile::new(Platform::MacOS, 26, 0)
    }

    fn empty_body() -> Body {
        let mut b = Body::new(0);
        b.num_blocks = 1;
        b.push(Inst::Ret(None));
        b
    }

    fn built<'a>(name: &'a str, params: Vec<Ty>) -> BuiltKernel<'a> {
        BuiltKernel {
            name,
            param_types: params,
            args: &[],
            body: empty_body(),
            pinned_types: vec![Ty::Int(64)],
        }
    }

    #[test]
    fn module_declaration_order_is_correct() {
        // globals, function prototypes, then kernels.
        let callees = vec![Callee {
            name: "@air.fast_exp.f32".into(),
            ret: Ty::Float,
            params: vec![Ty::Float],
            section: None,
        }];
        let globals = vec![("@scratch".to_string(), Ty::Float, 64u32)];
        let m = build_bitcode_module(
            &callees,
            vec![built("k", vec![Ty::ptr(Ty::Float, 1)])],
            &globals,
            &profile(),
        )
        .expect("module assembles");

        assert_eq!(m.globals.len(), 1);
        assert_eq!(m.funcs.len(), 2);
        assert!(m.funcs[0].is_proto, "function prototype first");
        assert!(!m.funcs[1].is_proto, "kernel second");
        assert_eq!(m.triple, profile().triple());
    }
}
