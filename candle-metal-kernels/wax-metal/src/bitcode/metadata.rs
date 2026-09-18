//! METADATA_BLOCK: the `!air.kernel` / argument-description metadata that makes Metal
//! recognise a function as a kernel at all.
//!
//! # Index space
//!
//! All metadata shares one index space, laid out in the order the block emits it:
//! strings first (all of them in a single `METADATA_STRINGS` record), then `VALUE`
//! records, then `NODE` records. So a node's index is `nstrings + nvalues + node_pos`.
//! Because that offset is only known once everything is collected, callers hold opaque
//! [`MdId`] handles and the writer resolves them at the end.
//!
//! Two encodings that differ and are easy to get backwards (both verified against
//! `metal-as` output):
//!
//! * `NODE` operands are stored as **index + 1**, with 0 meaning a null operand.
//! * `NAMED_NODE` operands are stored as **plain indices**.

use super::bitstream::BitWriter;
use super::{ConstPool, Ty, TypeTable};

const BLOCK_METADATA: u32 = 15;

const METADATA_CODE_VALUE: u32 = 2;
const METADATA_CODE_NODE: u32 = 3;
const METADATA_CODE_DISTINCT_NODE: u32 = 5;
const METADATA_CODE_NAME: u32 = 4;
const METADATA_CODE_NAMED_NODE: u32 = 10;
const METADATA_CODE_STRINGS: u32 = 35;

/// Handle to a metadata item. Resolved to a real index when the block is written.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MdId(Slot);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    Str(u32),
    Val(u32),
    Node(u32),
}

/// Collects a module's metadata.
#[derive(Default)]
pub struct MetadataBuilder {
    strings: Vec<String>,
    /// (type, absolute module value id)
    values: Vec<(Ty, u32)>,
    /// `None` operands are nulls.
    nodes: Vec<Vec<Option<MdId>>>,
    /// Parallel to `nodes`: whether each one is `distinct` (not uniqued). A
    /// self-referential node has to be distinct, since uniquing cannot hash a cycle.
    distinct: Vec<bool>,
    named: Vec<(String, Vec<MdId>)>,
}

impl MetadataBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// `!"text"`, deduplicated.
    pub fn string(&mut self, s: &str) -> MdId {
        if let Some(i) = self.strings.iter().position(|x| x == s) {
            return MdId(Slot::Str(i as u32));
        }
        self.strings.push(s.to_string());
        MdId(Slot::Str((self.strings.len() - 1) as u32))
    }

    /// A typed value operand such as `i32 7`, referring to an already-interned
    /// module-level constant or function by its absolute value id.
    pub fn value(&mut self, ty: Ty, value_id: u32) -> MdId {
        if let Some(i) = self
            .values
            .iter()
            .position(|(t, v)| *t == ty && *v == value_id)
        {
            return MdId(Slot::Val(i as u32));
        }
        self.values.push((ty, value_id));
        MdId(Slot::Val((self.values.len() - 1) as u32))
    }

    /// `!{ ... }`. Operands may be null.
    pub fn node(&mut self, ops: Vec<Option<MdId>>) -> MdId {
        self.nodes.push(ops);
        self.distinct.push(false);
        MdId(Slot::Node((self.nodes.len() - 1) as u32))
    }

    /// Reserve an empty `distinct` node whose operands are filled in later with
    /// [`set_node`](Self::set_node). This is what `!llvm.loop` needs: its first operand
    /// is the node itself, so the handle must exist before the operand list does.
    pub fn distinct_node(&mut self) -> MdId {
        self.nodes.push(Vec::new());
        self.distinct.push(true);
        MdId(Slot::Node((self.nodes.len() - 1) as u32))
    }

    /// Fill in a node reserved by [`distinct_node`](Self::distinct_node).
    pub fn set_node(&mut self, id: MdId, ops: &[MdId]) {
        let Slot::Node(i) = id.0 else {
            panic!("set_node on a non-node metadata handle");
        };
        self.nodes[i as usize] = ops.iter().map(|m| Some(*m)).collect();
    }

    /// Convenience for a node with no null operands.
    pub fn tuple(&mut self, ops: &[MdId]) -> MdId {
        self.node(ops.iter().map(|m| Some(*m)).collect())
    }

    /// `!name = !{ ... }`.
    pub fn named(&mut self, name: &str, ops: &[MdId]) {
        self.named.push((name.to_string(), ops.to_vec()));
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty() && self.values.is_empty() && self.nodes.is_empty()
    }

    pub(super) fn resolve(&self, id: MdId) -> u32 {
        let ns = self.strings.len() as u32;
        let nv = self.values.len() as u32;
        match id.0 {
            Slot::Str(i) => i,
            Slot::Val(i) => ns + i,
            Slot::Node(i) => ns + nv + i,
        }
    }

    /// The `METADATA_STRINGS` payload: a bitstream of VBR6 lengths padded to a word
    /// boundary, then the characters. The record carries the count and the byte offset
    /// at which the characters start.
    fn strings_blob(&self) -> (u64, Vec<u8>) {
        let mut sizes = BitWriter::new();
        for s in &self.strings {
            sizes.vbr(s.len() as u64, 6);
        }
        let mut blob = sizes.finish();
        let offset = blob.len() as u64;
        for s in &self.strings {
            blob.extend_from_slice(s.as_bytes());
        }
        (offset, blob)
    }

    pub(super) fn write(&self, w: &mut BitWriter, types: &TypeTable) {
        if self.is_empty() && self.named.is_empty() {
            return;
        }
        w.enter_block(BLOCK_METADATA, 4);

        if !self.strings.is_empty() {
            let (offset, blob) = self.strings_blob();
            let abbrev = w.define_strings_abbrev(METADATA_CODE_STRINGS);
            w.strings_record(abbrev, self.strings.len() as u64, offset, &blob);
        }

        for (ty, value_id) in &self.values {
            let ti = types.get(ty).expect("metadata value type interned");
            w.record(METADATA_CODE_VALUE, &[ti as u64, *value_id as u64]);
        }

        for (ops, distinct) in self.nodes.iter().zip(&self.distinct) {
            // Operands are index + 1 so that 0 can mean null.
            let encoded: Vec<u64> = ops
                .iter()
                .map(|o| o.map_or(0, |id| self.resolve(id) as u64 + 1))
                .collect();
            let code = if *distinct {
                METADATA_CODE_DISTINCT_NODE
            } else {
                METADATA_CODE_NODE
            };
            w.record(code, &encoded);
        }

        for (name, ops) in &self.named {
            w.record_str(METADATA_CODE_NAME, name);
            // Named-node operands are plain indices, unlike NODE's index + 1.
            let ids: Vec<u64> = ops.iter().map(|o| self.resolve(*o) as u64).collect();
            w.record(METADATA_CODE_NAMED_NODE, &ids);
        }

        w.end_block();
    }
}

/// METADATA_KIND_BLOCK: names the metadata kinds instructions can carry. The block is
/// self-describing - it defines the id/name mapping - so we only declare the kinds we
/// actually attach and can number them from zero.
#[derive(Default)]
pub struct MdKinds {
    names: Vec<String>,
}

const BLOCK_METADATA_KIND: u32 = 22;
const METADATA_CODE_KIND: u32 = 6;

impl MdKinds {
    /// Kind id for `name`, declaring it on first use.
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(i) = self.names.iter().position(|n| n == name) {
            return i as u32;
        }
        self.names.push(name.to_string());
        (self.names.len() - 1) as u32
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub(super) fn write(&self, w: &mut BitWriter) {
        if self.names.is_empty() {
            return;
        }
        w.enter_block(BLOCK_METADATA_KIND, 3);
        for (i, name) in self.names.iter().enumerate() {
            let mut ops = vec![i as u64];
            ops.extend(name.bytes().map(|b| b as u64));
            w.record(METADATA_CODE_KIND, &ops);
        }
        w.end_block();
    }
}

/// Builds the metadata every AIR kernel module needs: the module flags Metal expects,
/// the AIR/language versions, and `!air.kernel` describing the entry point.
///
/// `args` describes the kernel's parameters in order. Buffer arguments become
/// `air.buffer` nodes with an index and address space; the thread-position builtins
/// become their own node kinds. This mirrors what `metal-as` produces for a kernel
/// compiled from Metal source.
pub struct KernelMetadata<'a> {
    pub entry_value_id: u32,
    pub entry_fn_ptr_ty: Ty,
    pub args: &'a [KernelArg],
    pub air_version: (u32, u32, u32),
    pub language_version: (u32, u32, u32),
    pub limits: crate::profile::Limits,
    pub producer: &'a str,
}

/// How a buffer argument is accessed, which AIR records as a distinct key.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

impl Access {
    fn key(self) -> &'static str {
        match self {
            Access::Read => "air.read",
            Access::Write => "air.write",
            Access::ReadWrite => "air.read_write",
        }
    }
}

/// One kernel parameter, as AIR metadata describes it.
#[derive(Clone, Debug)]
pub enum KernelArg {
    /// A `device`/`constant` buffer: `air.buffer` with a binding index.
    Buffer {
        index: u32,
        /// 1 = device, 2 = constant.
        address_space: u32,
        access: Access,
        /// Element size and alignment in bytes.
        elem_size: u32,
        elem_align: u32,
        type_name: String,
        name: String,
    },
    /// A scalar passed in a constant buffer, e.g. a dynamic stride. Distinguished by
    /// carrying `air.buffer_size` instead of element size/alignment keys.
    ScalarBuffer {
        index: u32,
        byte_size: u32,
        type_name: String,
        name: String,
    },
    /// A position/id builtin, e.g. `air.thread_position_in_threadgroup`.
    Builtin {
        kind: String,
        type_name: String,
        name: String,
    },
}

impl KernelMetadata<'_> {
    /// Populate `md`, interning any integer constants needed into `consts`, and return
    /// the constants' value ids so the caller can keep the module consistent.
    ///
    /// `const_base` is the absolute value id of the first module constant (that is,
    /// the number of globals plus functions), since metadata refers to constants by
    /// absolute value id.
    pub fn build(
        &self,
        md: &mut MetadataBuilder,
        consts: &mut ConstPool,
        types: &mut TypeTable,
        const_base: u32,
    ) {
        let i32t = Ty::Int(32);
        types.intern(&i32t);
        let int = |md: &mut MetadataBuilder, consts: &mut ConstPool, v: i64| {
            let idx = consts.intern(super::Const::Int(i32t.clone(), v));
            md.value(i32t.clone(), const_base + idx)
        };

        // Per-argument nodes, in parameter order.
        let mut arg_nodes = Vec::new();
        for (i, a) in self.args.iter().enumerate() {
            let pos = int(md, consts, i as i64);
            let node = match a {
                KernelArg::Buffer {
                    index,
                    address_space,
                    access,
                    elem_size,
                    elem_align,
                    type_name,
                    name,
                } => {
                    let k_buffer = md.string("air.buffer");
                    let k_loc = md.string("air.location_index");
                    let k_rw = md.string(access.key());
                    let k_as = md.string("air.address_space");
                    let k_sz = md.string("air.arg_type_size");
                    let k_al = md.string("air.arg_type_align_size");
                    let k_tn = md.string("air.arg_type_name");
                    let k_an = md.string("air.arg_name");
                    let idx = int(md, consts, *index as i64);
                    let one = int(md, consts, 1);
                    let asp = int(md, consts, *address_space as i64);
                    let sz = int(md, consts, *elem_size as i64);
                    let al = int(md, consts, *elem_align as i64);
                    let tn = md.string(type_name);
                    let an = md.string(name);
                    md.tuple(&[
                        pos, k_buffer, k_loc, idx, one, k_rw, k_as, asp, k_sz, sz, k_al, al, k_tn,
                        tn, k_an, an,
                    ])
                }
                KernelArg::ScalarBuffer {
                    index,
                    byte_size,
                    type_name,
                    name,
                } => {
                    let k_buffer = md.string("air.buffer");
                    let k_size = md.string("air.buffer_size");
                    let k_loc = md.string("air.location_index");
                    let k_read = md.string("air.read");
                    let k_as = md.string("air.address_space");
                    let k_tn = md.string("air.arg_type_name");
                    let k_an = md.string("air.arg_name");
                    let sz = int(md, consts, *byte_size as i64);
                    let idx = int(md, consts, *index as i64);
                    let one = int(md, consts, 1);
                    let two = int(md, consts, 2);
                    let tn = md.string(type_name);
                    let an = md.string(name);
                    md.tuple(&[
                        pos, k_buffer, k_size, sz, k_loc, idx, one, k_read, k_as, two, k_tn, tn,
                        k_an, an,
                    ])
                }
                KernelArg::Builtin {
                    kind,
                    type_name,
                    name,
                } => {
                    let k = md.string(kind);
                    let k_tn = md.string("air.arg_type_name");
                    let k_an = md.string("air.arg_name");
                    let tn = md.string(type_name);
                    let an = md.string(name);
                    md.tuple(&[pos, k, k_tn, tn, k_an, an])
                }
            };
            arg_nodes.push(node);
        }

        let empty = md.node(vec![]);
        let arg_list = md.tuple(&arg_nodes);
        let entry = md.value(self.entry_fn_ptr_ty.clone(), self.entry_value_id);
        let kernel = md.tuple(&[entry, empty, arg_list]);
        md.named("air.kernel", &[kernel]);

        // Module flags Metal looks for. Behaviour 1 = error, 2 = warning, 7 = max.
        let mut flags = Vec::new();
        let flag = |md: &mut MetadataBuilder,
                    consts: &mut ConstPool,
                    behaviour: i64,
                    key: &str,
                    val: i64| {
            let b = int(md, consts, behaviour);
            let k = md.string(key);
            let v = int(md, consts, val);
            md.tuple(&[b, k, v])
        };
        flags.push(flag(md, consts, 1, "wchar_size", 4));
        flags.push(flag(md, consts, 7, "frame-pointer", 2));
        for (key, val) in self.limits.flags() {
            flags.push(flag(md, consts, 7, key, val));
        }
        md.named("llvm.module.flags", &flags);

        let (a0, a1, a2) = self.air_version;
        let av = [
            int(md, consts, a0 as i64),
            int(md, consts, a1 as i64),
            int(md, consts, a2 as i64),
        ];
        let air_v = md.tuple(&av);
        md.named("air.version", &[air_v]);

        let (l0, l1, l2) = self.language_version;
        let metal = md.string("Metal");
        let lv = [
            metal,
            int(md, consts, l0 as i64),
            int(md, consts, l1 as i64),
            int(md, consts, l2 as i64),
        ];
        let lang_v = md.tuple(&lv);
        md.named("air.language_version", &[lang_v]);

        let prod = md.string(self.producer);
        let ident = md.tuple(&[prod]);
        md.named("llvm.ident", &[ident]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_resolve_across_categories() {
        let mut md = MetadataBuilder::new();
        let s0 = md.string("a");
        let s1 = md.string("b");
        let v0 = md.value(Ty::Int(32), 100);
        let n0 = md.node(vec![Some(s0), None, Some(v0)]);
        // Two strings, then one value, then nodes.
        assert_eq!(md.resolve(s0), 0);
        assert_eq!(md.resolve(s1), 1);
        assert_eq!(md.resolve(v0), 2);
        assert_eq!(md.resolve(n0), 3);
    }

    #[test]
    fn strings_are_deduped() {
        let mut md = MetadataBuilder::new();
        let a = md.string("air.buffer");
        let b = md.string("air.buffer");
        assert_eq!(a, b);
        assert_eq!(md.strings.len(), 1);
    }

    /// The blob is VBR6 lengths, word-padded, then the characters; the offset points
    /// at the characters.
    #[test]
    fn strings_blob_offset_points_at_chars() {
        let mut md = MetadataBuilder::new();
        md.string("abc");
        md.string("defg");
        let (offset, blob) = md.strings_blob();
        assert_eq!(offset % 4, 0, "character data starts word aligned");
        assert_eq!(&blob[offset as usize..], b"abcdefg");
    }
}
