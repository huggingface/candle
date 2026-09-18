//! Writes LLVM bitcode in the dialect Apple's AIR reader accepts.
//!
//! Replaces `xcrun metal-as`, the last external tool in the pipeline - shelling out to
//! Xcode is impossible on iOS and in sandboxed apps.
//!
//! Dialect constraints, MEASURED:
//! * TYPED POINTERS. The reader is pre-opaque-pointer and wants `POINTER(pointee,
//!   addrspace)`. A stock LLVM 21 emits `OPAQUE_POINTER`; the library loads but then
//!   CRASHES MTLCompilerService at pipeline creation, so no upstream LLVM can stand in.
//! * `IDENTIFICATION` epoch 0, `MODULE_CODE_VERSION` 2 (selects string-table symbol names).
//!
//! Records are emitted unabbreviated: a reader must accept the long form, so this avoids
//! BLOCKINFO and abbreviation encodings for ~2x bytes on a small cached artifact. Operand
//! vectors for version-sensitive records (notably `MODULE_CODE_FUNCTION`) are replicated
//! from what `metal-as` emits, read with `llvm-bcanalyzer`, not derived from LLVM source.

// Some block IDs and record codes below are not referenced yet: they belong to the
// blocks still to be implemented (constants, metadata, attribute groups, function
// bodies), and are kept so the numbering lives in one place.
#![allow(dead_code)]

pub mod bitstream;
pub mod func;
pub mod metadata;

use bitstream::BitWriter;
use func::Body;
use metadata::{MdKinds, MetadataBuilder};
use std::collections::HashMap;

// Block IDs (LLVM StandardBlockIDs / bitcode block IDs).
const BLOCK_MODULE: u32 = 8;
const BLOCK_PARAMATTR: u32 = 9;
const BLOCK_PARAMATTR_GROUP: u32 = 10;
const BLOCK_CONSTANTS: u32 = 11;
const BLOCK_FUNCTION: u32 = 12;
const BLOCK_IDENTIFICATION: u32 = 13;
const BLOCK_TYPE: u32 = 17;
const BLOCK_STRTAB: u32 = 23;
const BLOCK_METADATA: u32 = 15;

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

/// An LLVM type, in the typed-pointer dialect.
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

/// Interns types and emits the TYPE_BLOCK.
///
/// Bitcode refers to types by index, and a composite type's operands must already
/// have indices, so `intern` registers components depth-first.
#[derive(Default)]
pub struct TypeTable {
    order: Vec<Ty>,
    index: HashMap<Ty, u32>,
}

impl TypeTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index of `ty`, registering it (and anything it contains) if new.
    pub fn intern(&mut self, ty: &Ty) -> u32 {
        if let Some(&i) = self.index.get(ty) {
            return i;
        }
        // Components first: their indices are operands of this type's record.
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

    /// Index of an already-interned type.
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

/// Concatenated symbol names. `MODULE_CODE_VERSION` 2 means every symbol names
/// itself by (offset, length) into this table rather than via a value symbol table.
#[derive(Default)]
pub struct StrTab {
    buf: String,
}

impl StrTab {
    /// Offset and length of `s`, appending it if not already present.
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
        // Must be a real blob: LLVM's reader requires STRTAB_BLOB to carry blob data,
        // not per-byte operands.
        let abbrev = w.define_blob_abbrev(STRTAB_CODE_BLOB);
        w.blob_record(abbrev, self.buf.as_bytes());
        w.end_block();
    }
}

/// LLVM `Attribute::AttrKind` numbers, as they appear in bitcode.
///
/// These values are LLVM-version-specific (the enum is generated, and inserting a new
/// attribute renumbers everything after it), so they are read out of what `metal-as`
/// itself emits rather than taken from any LLVM header. Identified by matching group
/// membership against the textual `attributes #N = { ... }` in the same module:
/// the definition carries `{convergent nounwind}` and uses the group `{43, 18}`, while
/// intrinsic declarations carry `{convergent nounwind willreturn}` and use
/// `{43, 18, 61}`.
pub mod attr {
    pub const NOUNWIND: u32 = 18;
    pub const CONVERGENT: u32 = 43;
    pub const WILLRETURN: u32 = 61;
}

/// PARAMATTR_GROUP applies to the function itself rather than a parameter.
const ATTR_IDX_FUNCTION: u64 = 0xFFFF_FFFF;

// PARAMATTR / PARAMATTR_GROUP record codes.
const PARAMATTR_CODE_ENTRY: u32 = 2;
const PARAMATTR_GRP_CODE_ENTRY: u32 = 3;

// CONSTANTS_BLOCK record codes.
const CST_CODE_SETTYPE: u32 = 1;
const CST_CODE_NULL: u32 = 2;
const CST_CODE_UNDEF: u32 = 3;
const CST_CODE_INTEGER: u32 = 4;
const CST_CODE_FLOAT: u32 = 6;
const CST_CODE_DATA: u32 = 22;

/// A module-level constant.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Const {
    /// An integer of the given type.
    Int(Ty, i64),
    /// A zeroinitializer of the given type.
    Null(Ty),
    /// `undef` of the given type - what an uninitialized threadgroup global uses.
    Undef(Ty),
    /// A float constant by raw bit pattern, at the type's width. Bits rather than an
    /// `f32` so f16/bf16 values (including infinities) stay exact.
    Float(Ty, u64),
    /// A dense array or vector of int/fp elements by raw bit pattern, e.g.
    /// `[2 x i32] [i32 26, i32 2]` or `<2 x i64> <i64 128, i64 8>`. This is LLVM's
    /// ConstantDataSequential encoding, which spans both shapes.
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

/// Module-level constants, in emission order. Metadata operands such as
/// `i32 7` reference these by value index, so the pool has to be built before the
/// metadata block is written.
#[derive(Default)]
pub struct ConstPool {
    items: Vec<Const>,
}

impl ConstPool {
    /// Index of `c` in the module's value table, interning it if new. The returned
    /// index is relative to the start of the constant pool; callers add the number of
    /// preceding values (globals and functions) to get an absolute value id.
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

    /// The types of every constant in the pool - all of which must be interned before
    /// the pool is written, since SETTYPE names them.
    pub fn types(&self) -> Vec<Ty> {
        self.items.iter().map(|c| c.ty().clone()).collect()
    }

    fn write(&self, w: &mut BitWriter, types: &TypeTable) {
        if self.items.is_empty() {
            return;
        }
        w.enter_block(BLOCK_CONSTANTS, 4);
        // SETTYPE is sticky: it applies to every following record until changed, so
        // only emit it when the type actually differs from the previous constant.
        let mut cur: Option<&Ty> = None;
        for c in &self.items {
            let ty = c.ty();
            if cur != Some(ty) {
                let ti = types.get(ty).expect("constant type interned");
                w.record(CST_CODE_SETTYPE, &[ti as u64]);
                cur = Some(ty);
            }
            match c {
                Const::Null(_) => w.record(CST_CODE_NULL, &[]),
                Const::Undef(_) => w.record(CST_CODE_UNDEF, &[]),
                // The bit pattern is stored directly; its width comes from SETTYPE.
                Const::Float(_, bits) => w.record(CST_CODE_FLOAT, &[*bits]),
                Const::Int(_, v) => w.record(CST_CODE_INTEGER, &[encode_signed(*v)]),
                Const::DataArray(_, vals) => w.record(CST_CODE_DATA, vals),
            }
        }
        w.end_block();
    }
}

/// LLVM's signed-integer record encoding: magnitude shifted up one place with the
/// sign in the low bit, so small values of either sign stay short under VBR.
fn encode_signed(v: i64) -> u64 {
    if v < 0 {
        (v.unsigned_abs() << 1) | 1
    } else {
        (v as u64) << 1
    }
}

/// A module-scope variable. Threadgroup scratch is the reason this exists:
/// `@_tg_user_scratch = internal unnamed_addr addrspace(3) global [N x float] undef`.
#[derive(Clone, Debug)]
pub struct Global {
    pub name: String,
    /// The *value* type (e.g. `[2112 x float]`), not a pointer to it.
    pub ty: Ty,
    pub address_space: u32,
    pub align: u32,
    /// Index in the module constant pool of the initializer, if any.
    pub init: Option<u32>,
    /// 3 = internal (what threadgroup scratch uses), 0 = external.
    pub linkage: u32,
    pub unnamed_addr: bool,
}

/// A function declaration or definition at module scope.
#[derive(Clone, Debug)]
pub struct FuncDecl {
    pub name: String,
    /// The function type (must be `Ty::Func`).
    pub ty: Ty,
    /// False for a `define`, true for a `declare`.
    pub is_proto: bool,
    /// Index into the PARAMATTR list (0 = none).
    pub attr_id: u32,
    /// 1-based index into the module section table, 0 for none. MPP's
    /// `__tensorops_impl_matmul2d_*` prototypes need `section
    /// "air.externally_defined"`, which is how the driver knows to supply their
    /// bodies at pipeline-creation time.
    pub section: u32,
}

/// A module being assembled for bitcode emission.
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
    pub funcs: Vec<FuncDecl>,
    /// Section names, in emission order; `FuncDecl::section` indexes this 1-based.
    sections: Vec<String>,
    /// Bodies for the non-proto functions, in the order they were declared.
    bodies: Vec<(usize, Body)>,
    /// Attribute groups, in emission order. Each is (group id, attribute kinds).
    attr_groups: Vec<(u32, Vec<u32>)>,
    /// Attribute lists; entry i (1-based for callers) names one group.
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

    /// Attribute-list id (1-based, as `FuncDecl::attr_id` wants) for `kinds`,
    /// creating the group and list on first use.
    pub fn attr_list(&mut self, kinds: &[u32]) -> u32 {
        if let Some(i) = self.attr_groups.iter().position(|(_, k)| k == kinds) {
            return self.attr_lists[i];
        }
        let gid = self.attr_groups.len() as u32 + 1;
        self.attr_groups.push((gid, kinds.to_vec()));
        self.attr_lists.push(gid);
        gid
    }

    /// `{convergent nounwind}` - what a kernel definition carries. `convergent` is
    /// load-bearing: without it the optimizer may move barriers and SIMD intrinsics
    /// across control flow and cross-SIMD reductions silently produce wrong results.
    pub fn kernel_attrs(&mut self) -> u32 {
        self.attr_list(&[attr::CONVERGENT, attr::NOUNWIND])
    }

    /// `{convergent nounwind willreturn}` - what AIR intrinsic declarations carry.
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
                // 0 marks a well-known enum attribute; the kind number follows.
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
        self.types.intern(&ty);
        self.funcs.push(FuncDecl {
            name: name.to_string(),
            ty,
            is_proto,
            attr_id,
            section: 0,
        });
    }

    /// Like [`Module::declare`], but placing the symbol in `section`.
    pub fn declare_in_section(
        &mut self,
        name: &str,
        ty: Ty,
        is_proto: bool,
        attr_id: u32,
        section: &str,
    ) {
        self.declare(name, ty, is_proto, attr_id);
        let id = self.section_id(section);
        self.funcs.last_mut().expect("just pushed").section = id;
    }

    /// 1-based id of `name` in the module section table, adding it if new.
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

    /// Declare a threadgroup global: an `undef`-initialised value in address space 3.
    /// `count > 1` makes it an array; `count == 1` a plain scalar, which is how the
    /// cross-SIMD reduce's single-value slot is declared.
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

    /// Declare threadgroup scratch: an `undef`-initialised array in address space 3.
    /// Returns its index in `globals` (which is also its module value id, since
    /// globals come first in the value table).
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

    /// Absolute value id of the first module-level constant. The module value table is
    /// globals, then functions, then constants - so adding a global shifts every
    /// function's id, which is why callers must not cache these.
    pub fn const_base(&self) -> u32 {
        (self.globals.len() + self.funcs.len()) as u32
    }

    /// Absolute value id of the function declared at `index`.
    pub fn func_value_id(&self, index: u32) -> u32 {
        self.globals.len() as u32 + index
    }

    /// Emit the whole file: wrapper, IDENTIFICATION, MODULE, STRTAB.
    pub fn to_bitcode(&mut self) -> Vec<u8> {
        let body = self.write_stream();
        // 0x0b17c0de wrapper: what metal-as emits for this target. Offset is the
        // header size, CPU type is left unset.
        let mut out = Vec::with_capacity(body.len() + 20);
        out.extend_from_slice(&0x0b17_c0deu32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&20u32.to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&0xffff_ffffu32.to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn write_stream(&mut self) -> Vec<u8> {
        // Names must be interned before the FUNCTION records reference them.
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
        w.fixed(b'B' as u32, 8);
        w.fixed(b'C' as u32, 8);
        w.fixed(0x0, 4);
        w.fixed(0xC, 4);
        w.fixed(0xE, 4);
        w.fixed(0xD, 4);

        w.enter_block(BLOCK_IDENTIFICATION, 5);
        w.record_str(IDENTIFICATION_CODE_STRING, "");
        w.record(IDENTIFICATION_CODE_EPOCH, &[0]);
        w.end_block();

        w.enter_block(BLOCK_MODULE, 3);
        w.record(MODULE_CODE_VERSION, &[2]);
        self.types.write(&mut w);
        self.write_attrs(&mut w);
        w.record_str(MODULE_CODE_TRIPLE, &self.triple);
        w.record_str(MODULE_CODE_DATALAYOUT, &self.datalayout);
        if !self.source_filename.is_empty() {
            w.record_str(MODULE_CODE_SOURCE_FILENAME, &self.source_filename);
        }
        let const_base = (self.globals.len() + self.funcs.len()) as u32;
        for (g, (off, len)) in self.globals.iter().zip(&global_slots) {
            let ti = self.types.get(&g.ty).expect("global type interned");
            // The initializer is a forward reference into the constants block, stored
            // as id + 1 so that 0 can mean "no initializer".
            let init_id = g.init.map_or(0, |i| (const_base + i) as u64 + 1);
            w.record(
                MODULE_CODE_GLOBALVAR,
                &[
                    *off,
                    *len,
                    ti as u64,
                    // isConstant | explicitType << 1 | addrspace << 2
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
        // SECTIONNAME records must precede the FUNCTION records that index them.
        for s in &self.sections {
            w.record_str(MODULE_CODE_SECTIONNAME, s);
        }
        for (f, (off, len)) in self.funcs.iter().zip(&name_slots) {
            let fty = self.types.get(&f.ty).expect("function type interned");
            // Layout replicated from metal-as output (21 operands). Everything past
            // the attribute id is a default; op19 is 14 as that tool emits it.
            w.record(
                MODULE_CODE_FUNCTION,
                &[
                    *off,
                    *len,
                    fty as u64,
                    0, // calling convention
                    f.is_proto as u64,
                    0, // linkage: external
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
        self.consts.write(&mut w, &self.types);
        self.md_kinds.write(&mut w);
        self.md.write(&mut w, &self.types);
        // Bodies come last, in declaration order; the reader pairs them with the
        // non-proto FUNCTION records in order.
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
        // A cut-down datalayout: these tests exercise the encoder, not the profile.
        Module::new(
            crate::profile::TargetProfile::host().triple(),
            "e-p:64:64:64-n8:16:32",
        )
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

    /// The string table must land in the file as raw bytes. It is emitted through a
    /// blob abbreviation; if it ever regresses to per-byte record operands those
    /// bytes get VBR6-packed and no longer appear verbatim - and LLVM's reader
    /// rejects the module outright ("Invalid record").
    #[test]
    fn strtab_is_stored_as_a_verbatim_blob() {
        let mut m = air_module();
        m.declare("some_kernel_name", Ty::func(Ty::Void, vec![]), true, 0);
        let bc = m.to_bitcode();
        let needle = b"some_kernel_name";
        assert!(
            bc.windows(needle.len()).any(|w| w == needle),
            "strtab bytes must appear verbatim (blob), not VBR-packed"
        );
    }

    /// The file must start with the wrapper then the 'BC' magic, and the stream
    /// length recorded in the wrapper must match the payload.
    #[test]
    fn emits_wrapper_and_magic() {
        let mut m = air_module();
        m.declare(
            "k",
            Ty::func(Ty::Void, vec![Ty::ptr(Ty::Float, 1)]),
            false,
            0,
        );
        let bc = m.to_bitcode();
        assert_eq!(&bc[0..4], &0x0b17_c0deu32.to_le_bytes());
        let off = u32::from_le_bytes(bc[8..12].try_into().unwrap()) as usize;
        let size = u32::from_le_bytes(bc[12..16].try_into().unwrap()) as usize;
        assert_eq!(off, 20);
        assert_eq!(off + size, bc.len(), "wrapper size covers the stream");
        assert_eq!(&bc[20..24], &[b'B', b'C', 0xC0, 0xDE], "bitcode magic");
        assert_eq!(bc.len() % 4, 0, "stream is word aligned");
    }
}
