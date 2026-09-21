//! LLVM bitcode ala AIR.

pub mod bitstream;
pub mod func;
pub mod metadata;

use bitstream::BitWriter;
use func::Body;
use metadata::{MdKinds, MetadataBuilder};
use std::collections::HashMap;

// Block ids (LLVM StandardBlockIDs / bitcode block ids).
const BLOCK_MODULE: u32 = 8;
const BLOCK_PARAMATTR: u32 = 9;
const BLOCK_PARAMATTR_GROUP: u32 = 10;
const BLOCK_CONSTANTS: u32 = 11;
const BLOCK_FUNCTION: u32 = 12;
const BLOCK_IDENTIFICATION: u32 = 13;
const BLOCK_METADATA: u32 = 15;
const BLOCK_METADATA_ATTACHMENT: u32 = 16;
const BLOCK_TYPE: u32 = 17;
const BLOCK_METADATA_KIND: u32 = 22;
const BLOCK_STRTAB: u32 = 23;

// IDENTIFICATION_BLOCK record codes.
const IDENTIFICATION_CODE_STRING: u32 = 1;
const IDENTIFICATION_CODE_EPOCH: u32 = 2;

// MODULE_BLOCK record codes.
const MODULE_CODE_VERSION: u32 = 1;
const MODULE_CODE_TRIPLE: u32 = 2;
const MODULE_CODE_DATALAYOUT: u32 = 3;
const MODULE_CODE_SECTIONNAME: u32 = 5;
const MODULE_CODE_GLOBALVAR: u32 = 7;
const MODULE_CODE_FUNCTION: u32 = 8;
const MODULE_CODE_SOURCE_FILENAME: u32 = 16;

// TYPE_BLOCK record codes.
const TYPE_CODE_NUMENTRY: u32 = 1;
const TYPE_CODE_VOID: u32 = 2;
const TYPE_CODE_FLOAT: u32 = 3;
const TYPE_CODE_DOUBLE: u32 = 4;
const TYPE_CODE_LABEL: u32 = 5;
const TYPE_CODE_INTEGER: u32 = 7;
const TYPE_CODE_POINTER: u32 = 8;
const TYPE_CODE_HALF: u32 = 10;
const TYPE_CODE_ARRAY: u32 = 11;
const TYPE_CODE_VECTOR: u32 = 12;
const TYPE_CODE_METADATA: u32 = 16;
const TYPE_CODE_FUNCTION: u32 = 21;
const TYPE_CODE_BFLOAT: u32 = 23;

const STRTAB_CODE_BLOB: u32 = 1;

/// LLVM types, with typed pointers.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    Void,
    Label,
    Metadata,
    Int(u32),
    Half,
    BFloat,
    Float,
    Double,
    /// `pointee addrspace(space)*`
    Ptr(Box<Ty>, u32),
    Vector(u32, Box<Ty>),
    Array(u64, Box<Ty>),
    Func {
        ret: Box<Ty>,
        params: Vec<Ty>,
        vararg: bool,
    },
}

impl Ty {
    pub fn ptr(pointee: Ty, space: u32) -> Ty {
        Ty::Ptr(Box::new(pointee), space)
    }
    pub fn vec(n: u32, elem: Ty) -> Ty {
        Ty::Vector(n, Box::new(elem))
    }
    pub fn func(ret: Ty, params: Vec<Ty>) -> Ty {
        Ty::Func {
            ret: Box::new(ret),
            params,
            vararg: false,
        }
    }
}

/// Type table interns types and emits TYPE_BLOCK.
#[derive(Default)]
pub struct TypeTable {
    order: Vec<Ty>,
    index: HashMap<Ty, u32>,
}

impl TypeTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or insert `ty`, returning type table index.
    pub fn intern(&mut self, ty: &Ty) -> u32 {
        if let Some(&i) = self.index.get(ty) {
            return i;
        }
        match ty {
            Ty::Ptr(p, _) => {
                self.intern(p);
            }
            Ty::Vector(_, e) | Ty::Array(_, e) => {
                self.intern(e);
            }
            Ty::Func { ret, params, .. } => {
                self.intern(ret);
                for p in params {
                    self.intern(p);
                }
            }
            _ => {}
        }
        if let Some(&i) = self.index.get(ty) {
            return i;
        }
        let i = self.order.len() as u32;
        self.order.push(ty.clone());
        self.index.insert(ty.clone(), i);
        i
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Table index of a type.
    pub fn get(&self, ty: &Ty) -> Option<u32> {
        self.index.get(ty).copied()
    }

    fn write(&self, w: &mut BitWriter) {
        w.enter_block(BLOCK_TYPE, 4);
        w.record(TYPE_CODE_NUMENTRY, &[self.order.len() as u64]);
        for ty in &self.order {
            match ty {
                Ty::Void => w.record(TYPE_CODE_VOID, &[]),
                Ty::Label => w.record(TYPE_CODE_LABEL, &[]),
                Ty::Metadata => w.record(TYPE_CODE_METADATA, &[]),
                Ty::Half => w.record(TYPE_CODE_HALF, &[]),
                Ty::BFloat => w.record(TYPE_CODE_BFLOAT, &[]),
                Ty::Float => w.record(TYPE_CODE_FLOAT, &[]),
                Ty::Double => w.record(TYPE_CODE_DOUBLE, &[]),
                Ty::Int(bits) => w.record(TYPE_CODE_INTEGER, &[*bits as u64]),
                Ty::Ptr(p, space) => {
                    let pi = self.index[p.as_ref()] as u64;
                    w.record(TYPE_CODE_POINTER, &[pi, *space as u64]);
                }
                Ty::Vector(n, e) => {
                    let ei = self.index[e.as_ref()] as u64;
                    w.record(TYPE_CODE_VECTOR, &[*n as u64, ei]);
                }
                Ty::Array(n, e) => {
                    let ei = self.index[e.as_ref()] as u64;
                    w.record(TYPE_CODE_ARRAY, &[*n, ei]);
                }
                Ty::Func {
                    ret,
                    params,
                    vararg,
                } => {
                    let mut ops = vec![*vararg as u64, self.index[ret.as_ref()] as u64];
                    for p in params {
                        ops.push(self.index[p] as u64);
                    }
                    w.record(TYPE_CODE_FUNCTION, &ops);
                }
            }
        }
        w.end_block();
    }
}

/// Symbol names.
#[derive(Default)]
pub struct StrTab {
    buf: String,
}

impl StrTab {
    /// Get or insert symbol name, returning (`offset`, `length`).
    pub fn intern(&mut self, s: &str) -> (u64, u64) {
        if let Some(pos) = self.buf.find(s) {
            return (pos as u64, s.len() as u64);
        }
        let off = self.buf.len() as u64;
        self.buf.push_str(s);
        (off, s.len() as u64)
    }

    fn write(&self, w: &mut BitWriter) {
        w.enter_block(BLOCK_STRTAB, 3);
        let abbrev = w.define_blob_abbr(STRTAB_CODE_BLOB);
        w.blob_record(abbrev, self.buf.as_bytes());
        w.end_block();
    }
}

/// LLVM `Attribute::AttrKind` as bitcode values., as they appear in bitcode.
///
/// Set as `attributes #N = { ... }` the module, where `{convergent nounwind}` becomes `{43, 18}`,
/// and `{convergent nounwind willreturn}` becomes `{43, 18, 61}`.
pub mod attr {
    pub const NOUNWIND: u32 = 18;
    pub const CONVERGENT: u32 = 43;
    pub const WILLRETURN: u32 = 61;
}

const ATTR_IDX_FUNCTION: u64 = 0xFFFF_FFFF;
const PARAMATTR_CODE_ENTRY: u32 = 2;
const PARAMATTR_GRP_CODE_ENTRY: u32 = 3;

// CONSTANTS_BLOCK  bitcodes.
const CST_CODE_SETTYPE: u32 = 1;
const CST_CODE_NULL: u32 = 2;
const CST_CODE_UNDEF: u32 = 3;
const CST_CODE_INTEGER: u32 = 4;
const CST_CODE_FLOAT: u32 = 6;
const CST_CODE_DATA: u32 = 22;

/// Module level constant.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Const {
    /// Integer of given type.
    Int(Ty, i64),
    /// Null of given type.
    Null(Ty),
    /// `undef` of given type. Used, for example, by uninitialized threadgroup globals.
    Undef(Ty),
    /// Float constant as raw bits, at type's width.
    Float(Ty, u64),
    /// Array/vector of ints/floats as raw bits.
    /// For example `[2 x f32] [f32 26, f32 2]`, or `<2 x i32> <i32 128, i32 8>`.
    /// LLVM's ConstantDataSequential encoding.
    DataArray(Ty, Vec<u64>),
}

impl Const {
    fn ty(&self) -> &Ty {
        match self {
            Const::Int(t, _)
            | Const::Null(t)
            | Const::Undef(t)
            | Const::Float(t, _)
            | Const::DataArray(t, _) => t,
        }
    }
}

/// Pool of module constants.
#[derive(Default, Debug)]
pub struct ConstPool {
    items: Vec<Const>,
}

impl ConstPool {
    /// Get or insert constant into the pool, returning index.
    pub fn intern(&mut self, c: Const) -> u32 {
        if let Some(i) = self.items.iter().position(|x| *x == c) {
            return i as u32;
        }
        self.items.push(c);
        (self.items.len() - 1) as u32
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// All types present in the pool.
    // TODO: dedup?
    pub fn types(&self) -> Vec<Ty> {
        self.items.iter().map(|c| c.ty().clone()).collect()
    }

    fn write(&self, w: &mut BitWriter, types: &TypeTable) {
        if self.items.is_empty() {
            return;
        }
        w.enter_block(BLOCK_CONSTANTS, 4);
        let mut cur: Option<&Ty> = None;
        for c in &self.items {
            let ty = c.ty();
            // Only change `SETTYPE` when type changes.
            if cur != Some(ty) {
                let ti = types.get(ty).expect("interned: Module::check_types");
                w.record(CST_CODE_SETTYPE, &[ti as u64]);
                cur = Some(ty);
            }
            match c {
                Const::Null(_) => w.record(CST_CODE_NULL, &[]),
                Const::Undef(_) => w.record(CST_CODE_UNDEF, &[]),
                Const::Float(_, bits) => w.record(CST_CODE_FLOAT, &[*bits]),
                Const::Int(_, v) => w.record(CST_CODE_INTEGER, &[encode_signed(*v)]),
                Const::DataArray(_, vals) => w.record(CST_CODE_DATA, vals),
            }
        }
        w.end_block();
    }
}

/// LLVM signed integer encoding.
fn encode_signed(v: i64) -> u64 {
    if v < 0 {
        (v.unsigned_abs() << 1) | 1
    } else {
        (v as u64) << 1
    }
}

/// Global variable type. Required for threadgroup scratch.
/// `@_tg_user_scratch = internal unnamed_addr addrspace(3) global [N x float] undef`.
#[derive(Clone, Debug)]
pub struct Global {
    pub name: String,
    pub ty: Ty,
    pub address_space: u32,
    pub align: u32,
    /// Constant pool index of initializer.
    pub init: Option<u32>,
    /// internal = 3, external = 0.
    pub linkage: u32,
    pub unnamed_addr: bool,
}

// TODO: Use this instead of bool flag in `Function`?
// enum FunctionType {
//     Define,
//     Declare,
// }

/// Module function (define/declare).
#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    // TODO: Must be `Ty::Func`. Perhaps `Func {ret, params, varag}` should be its own type?
    pub ty: Ty,
    /// `declare`/`define` function.
    pub is_proto: bool,
    /// Index into the `PARAMATTR` list (0 means none).
    pub attr_id: u32,
    /// Index into module section table.
    /// This lets us supply e.g. `section "air.externally_defined"` for functions that need them.
    pub section: u32,
}

/// Module state for bitcode emission.
pub struct Module {
    pub triple: String,
    pub datalayout: String,
    pub source_filename: String,
    pub types: TypeTable,
    pub strtab: StrTab,
    pub consts: ConstPool,
    pub md: MetadataBuilder,
    pub md_kinds: MdKinds,
    pub globals: Vec<Global>,
    pub funcs: Vec<Function>,
    sections: Vec<String>,
    /// Function bodies `(id, body)`.
    bodies: Vec<(usize, Body)>,
    /// Attribute groups `(group id, attribute kinds)`.
    attr_groups: Vec<(u32, Vec<u32>)>,
    /// Attribute lists. One entry per attribute group.
    attr_lists: Vec<u32>,
}

impl Module {
    pub fn new(triple: impl Into<String>, datalayout: impl Into<String>) -> Self {
        Self {
            triple: triple.into(),
            datalayout: datalayout.into(),
            source_filename: String::new(),
            types: TypeTable::new(),
            strtab: StrTab::default(),
            consts: ConstPool::default(),
            md: MetadataBuilder::new(),
            md_kinds: MdKinds::default(),
            globals: Vec::new(),
            funcs: Vec::new(),
            sections: Vec::new(),
            bodies: Vec::new(),
            attr_groups: Vec::new(),
            attr_lists: Vec::new(),
        }
    }

    /// Get or insert attribute kinds, returning attribute group id.
    pub fn attr_list(&mut self, kinds: &[u32]) -> u32 {
        if let Some(i) = self.attr_groups.iter().position(|(_, k)| k == kinds) {
            return self.attr_lists[i];
        }
        let gid = self.attr_groups.len() as u32 + 1;
        self.attr_groups.push((gid, kinds.to_vec()));
        self.attr_lists.push(gid);
        gid
    }

    /// Default attributes for kernels: `{convergent nounwind}`.
    /// Note that without  `convergent` the optimizer may move barriers and SIMD intrinsics across
    /// control flow, so that e.g. cross-SIMD reductions could silently produce incorrect results.
    pub fn kernel_attrs(&mut self) -> u32 {
        self.attr_list(&[attr::CONVERGENT, attr::NOUNWIND])
    }

    /// Default attributes ofr AIR intrinsic declarations: `{convergent nounwind willreturn}`.
    pub fn intrinsic_attrs(&mut self) -> u32 {
        self.attr_list(&[attr::CONVERGENT, attr::NOUNWIND, attr::WILLRETURN])
    }

    fn write_attrs(&self, w: &mut BitWriter) {
        if self.attr_groups.is_empty() {
            return;
        }
        w.enter_block(BLOCK_PARAMATTR_GROUP, 3);
        for (gid, kinds) in &self.attr_groups {
            let mut ops = vec![*gid as u64, ATTR_IDX_FUNCTION];
            for k in kinds {
                ops.push(0);
                ops.push(*k as u64);
            }
            w.record(PARAMATTR_GRP_CODE_ENTRY, &ops);
        }
        w.end_block();

        w.enter_block(BLOCK_PARAMATTR, 3);
        for gid in &self.attr_lists {
            w.record(PARAMATTR_CODE_ENTRY, &[*gid as u64]);
        }
        w.end_block();
    }

    pub fn declare(&mut self, name: &str, ty: Ty, is_proto: bool, attr_id: u32) {
        self.declare_at(name, ty, is_proto, attr_id);
    }

    /// [`Module::declare`] and returns current function index.
    fn declare_at(&mut self, name: &str, ty: Ty, is_proto: bool, attr_id: u32) -> usize {
        self.types.intern(&ty);
        self.funcs.push(Function {
            name: name.to_string(),
            ty,
            is_proto,
            attr_id,
            section: 0,
        });
        self.funcs.len() - 1
    }

    /// [`Module::declare`], with symbol placement in `section`.
    pub fn declare_in_section(
        &mut self,
        name: &str,
        ty: Ty,
        is_proto: bool,
        attr_id: u32,
        section: &str,
    ) {
        let at = self.declare_at(name, ty, is_proto, attr_id);
        let id = self.section_id(section);
        self.funcs[at].section = id;
    }

    /// Get or insert section `name` module section table, returning section id.
    fn section_id(&mut self, name: &str) -> u32 {
        if let Some(i) = self.sections.iter().position(|s| s == name) {
            return i as u32 + 1;
        }
        self.sections.push(name.to_string());
        self.sections.len() as u32
    }

    /// Attach a body to the function declared at `func_index`.
    pub fn set_body(&mut self, func_index: usize, body: Body) {
        for t in body.referenced_types() {
            self.types.intern(&t);
        }
        self.bodies.push((func_index, body));
    }

    /// Verify that every type required to write body is present in the type table.
    pub fn check_types(&self) -> Result<(), String> {
        let mut want: Vec<(String, Ty)> = Vec::new();
        for (idx, body) in &self.bodies {
            let whose = format!("function {idx}");
            want.extend(
                body.referenced_types()
                    .into_iter()
                    .map(|t| (whose.clone(), t)),
            );
        }
        want.extend(
            self.consts
                .types()
                .into_iter()
                .map(|t| ("module constant".to_string(), t)),
        );
        want.extend(
            self.md
                .value_types()
                .into_iter()
                .map(|t| ("metadata value".to_string(), t)),
        );
        want.extend(
            self.globals
                .iter()
                .map(|g| (format!("global `{}`", g.name), g.ty.clone())),
        );
        want.extend(
            self.funcs
                .iter()
                .map(|f| (format!("function `{}`", f.name), f.ty.clone())),
        );
        for (whose, t) in want {
            if self.types.get(&t).is_none() {
                return Err(format!("{whose}: type {t:?} is not interned"));
            }
        }
        Ok(())
    }

    /// Declare a threadgroup global.
    pub fn threadgroup_global(&mut self, name: &str, elem: Ty, count: u64) -> u32 {
        if count > 1 {
            return self.threadgroup_scratch(name, count, elem, 4);
        }
        self.types.intern(&elem);
        let init = self.consts.intern(Const::Undef(elem.clone()));
        self.globals.push(Global {
            name: name.to_string(),
            ty: elem,
            address_space: 3,
            align: 4,
            init: Some(init),
            linkage: 3,
            unnamed_addr: true,
        });
        (self.globals.len() - 1) as u32
    }

    /// Declare threadgroup scratch.
    pub fn threadgroup_scratch(&mut self, name: &str, elems: u64, elem: Ty, align: u32) -> u32 {
        let ty = Ty::Array(elems, Box::new(elem));
        self.types.intern(&ty);
        let init = self.consts.intern(Const::Undef(ty.clone()));
        self.globals.push(Global {
            name: name.to_string(),
            ty,
            address_space: 3,
            align,
            init: Some(init),
            linkage: 3,
            unnamed_addr: true,
        });
        (self.globals.len() - 1) as u32
    }

    /// Absolute value id of the first constant.
    pub fn const_base(&self) -> u32 {
        (self.globals.len() + self.funcs.len()) as u32
    }

    /// Absolute value id of the function declared at `index`.
    pub fn func_value_id(&self, index: u32) -> u32 {
        self.globals.len() as u32 + index
    }

    /// Emit the whole file.
    /// Wrapper, `IDENTIFICATION`, `MODULE`, `STRTAB`.
    pub fn to_bitcode(&mut self) -> Result<Vec<u8>, String> {
        self.check_types()?;
        let body = self.write_stream();
        // Emit "bitcode" (0x0b17c0de) magic and header info, then the body itself.
        let mut out = Vec::with_capacity(body.len() + 20);
        out.extend_from_slice(&0x0b17_c0deu32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&20u32.to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&0xffff_ffffu32.to_le_bytes());
        out.extend_from_slice(&body);
        Ok(out)
    }

    fn write_stream(&mut self) -> Vec<u8> {
        // Names must be interned before `FUNCTION` records reference them.
        let global_names: Vec<String> = self.globals.iter().map(|g| g.name.clone()).collect();
        let global_slots: Vec<(u64, u64)> =
            global_names.iter().map(|n| self.strtab.intern(n)).collect();
        let name_slots: Vec<(u64, u64)> = self
            .funcs
            .iter()
            .map(|f| f.name.clone())
            .collect::<Vec<_>>()
            .iter()
            .map(|n| self.strtab.intern(n))
            .collect();

        let mut w = BitWriter::new();

        // LLVM bitcode magic
        w.fixed(b'B' as u32, 8);
        w.fixed(b'C' as u32, 8);
        w.fixed(0x0, 4);
        w.fixed(0xC, 4);
        w.fixed(0xE, 4);
        w.fixed(0xD, 4);

        // Empty identification string and epoch matches Metal.
        w.enter_block(BLOCK_IDENTIFICATION, 5);
        w.record_str(IDENTIFICATION_CODE_STRING, "");
        w.record(IDENTIFICATION_CODE_EPOCH, &[0]);
        w.end_block();

        // Module block
        w.enter_block(BLOCK_MODULE, 3);
        // Same as Metal
        w.record(MODULE_CODE_VERSION, &[2]);
        // Write types, attributes, target triple, and data layout.
        self.types.write(&mut w);
        self.write_attrs(&mut w);
        w.record_str(MODULE_CODE_TRIPLE, &self.triple);
        w.record_str(MODULE_CODE_DATALAYOUT, &self.datalayout);
        // Write source filename if present.
        if !self.source_filename.is_empty() {
            w.record_str(MODULE_CODE_SOURCE_FILENAME, &self.source_filename);
        }
        // Calculate const base index
        let const_base = (self.globals.len() + self.funcs.len()) as u32;
        // Write globals
        for (g, (off, len)) in self.globals.iter().zip(&global_slots) {
            let ti = self
                .types
                .get(&g.ty)
                .expect("interned: Module::check_types");
            let init_id = g.init.map_or(0, |i| (const_base + i) as u64 + 1);
            w.record(
                MODULE_CODE_GLOBALVAR,
                &[
                    *off,
                    *len,
                    ti as u64,
                    // is_constant | explicit_type << 1 | addrspace << 2
                    1 << 1 | (g.address_space as u64) << 2,
                    init_id,
                    g.linkage as u64,
                    g.align.trailing_zeros() as u64 + 1,
                    0, // section
                    0, // visibility
                    0, // thread local
                    g.unnamed_addr as u64,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            );
        }
        // Sections recorded before the functions that reference them.
        for s in &self.sections {
            w.record_str(MODULE_CODE_SECTIONNAME, s);
        }
        for (f, (off, len)) in self.funcs.iter().zip(&name_slots) {
            let fty = self
                .types
                .get(&f.ty)
                .expect("interned: Module::check_types");
            // Same as metal-as.
            w.record(
                MODULE_CODE_FUNCTION,
                &[
                    *off,
                    *len,
                    fty as u64,
                    0, // calling convention
                    f.is_proto as u64,
                    0, // external linking
                    f.attr_id as u64,
                    0,                // alignment
                    f.section as u64, // section: 1-based, 0 = none
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            );
        }
        // Write constants
        self.consts.write(&mut w, &self.types);
        // Write metadata
        self.md_kinds.write(&mut w);
        self.md.write(&mut w, &self.types);

        // Write bodies, sorted. Reader pairs them with `FUNCTION` records.
        let func_base = const_base + self.consts.len() as u32;
        let mut ordered: Vec<&(usize, Body)> = self.bodies.iter().collect();
        ordered.sort_by_key(|(i, _)| *i);
        for (_, body) in ordered {
            body.write(
                &mut w,
                &self.types,
                &self.md,
                self.globals.len() as u32,
                const_base,
                func_base,
            );
        }
        w.end_block();

        self.strtab.write(&mut w);
        w.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn air_module() -> Module {
        // Example datalayout. Tests validate the encoder, not the profile.
        Module::new(
            crate::profile::TargetProfile::host().triple(),
            "e-p:64:64:64-n8:16:32",
        )
    }

    #[test]
    fn uninterned_global_type_errors_correctly() {
        let mut m = air_module();
        m.globals.push(Global {
            name: "@g".to_string(),
            ty: Ty::Double,
            address_space: 3,
            align: 4,
            init: None,
            linkage: 3,
            unnamed_addr: true,
        });
        let e = m
            .to_bitcode()
            .expect_err("an un-interned global type is refused");
        assert!(e.contains("@g"), "the error names the global: {e}");
    }

    #[test]
    fn type_table_interns_components_before_composites() {
        let mut t = TypeTable::new();
        let f = Ty::ptr(Ty::Float, 1);
        let i = t.intern(&f);
        // float must have been registered first, so it has the lower index.
        let fl = t.get(&Ty::Float).expect("float interned");
        assert!(fl < i, "pointee index {fl} must precede pointer index {i}");
        // Interning again is stable.
        assert_eq!(t.intern(&f), i);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn type_table_dedups_structurally_equal_types() {
        let mut t = TypeTable::new();
        let a = t.intern(&Ty::vec(3, Ty::Int(32)));
        let b = t.intern(&Ty::vec(3, Ty::Int(32)));
        assert_eq!(a, b);
        assert_eq!(t.len(), 2, "i32 and <3 x i32> only");
    }

    #[test]
    fn strtab_reuses_substrings() {
        let mut s = StrTab::default();
        let (o1, l1) = s.intern("kernel_a");
        let (o2, l2) = s.intern("kernel_a");
        assert_eq!((o1, l1), (o2, l2));
        assert_eq!(s.buf, "kernel_a");
    }

    #[test]
    fn signed_constants_encode_sign_in_the_low_bit() {
        assert_eq!(encode_signed(0), 0);
        assert_eq!(encode_signed(1), 2);
        assert_eq!(encode_signed(31), 62);
        assert_eq!(encode_signed(-7), 15);
        assert_eq!(encode_signed(-1), 3);
    }

    #[test]
    fn attr_lists_are_deduped_and_one_based() {
        let mut m = air_module();
        let a = m.kernel_attrs();
        let b = m.kernel_attrs();
        let c = m.intrinsic_attrs();
        assert_eq!(a, b, "same attribute set reuses its list");
        assert_ne!(a, c);
        assert_eq!(a, 1, "attribute list ids are 1-based (0 means none)");
        assert_eq!(c, 2);
    }

    #[test]
    fn const_pool_dedups() {
        let mut p = ConstPool::default();
        let a = p.intern(Const::Int(Ty::Int(32), 5));
        let b = p.intern(Const::Int(Ty::Int(32), 5));
        let c = p.intern(Const::Int(Ty::Int(32), 6));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn strtab_is_stored_as_a_verbatim_blob() {
        let mut m = air_module();
        m.declare("some_kernel_name", Ty::func(Ty::Void, vec![]), true, 0);
        let bc = m.to_bitcode().expect("the module's types are interned");
        let needle = b"some_kernel_name";
        assert!(
            bc.windows(needle.len()).any(|w| w == needle),
            "strtab bytes must appear verbatim (blob), not VBR-packed"
        );
    }

    #[test]
    fn emits_wrapper_and_magic() {
        let mut m = air_module();
        m.declare(
            "k",
            Ty::func(Ty::Void, vec![Ty::ptr(Ty::Float, 1)]),
            false,
            0,
        );
        let bc = m.to_bitcode().expect("module types are interned");
        assert_eq!(&bc[0..4], &0x0b17_c0deu32.to_le_bytes());
        let off = u32::from_le_bytes(bc[8..12].try_into().unwrap()) as usize;
        let size = u32::from_le_bytes(bc[12..16].try_into().unwrap()) as usize;
        assert_eq!(off, 20);
        assert_eq!(off + size, bc.len(), "wrapper size covers the stream");
        assert_eq!(&bc[20..24], &[b'B', b'C', 0xC0, 0xDE], "bitcode magic");
        assert_eq!(bc.len() % 4, 0, "stream is word aligned");
    }
}
