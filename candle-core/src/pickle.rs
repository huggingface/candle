//! Just enough pickle support to be able to read PyTorch checkpoints.
// This hardcodes objects that are required for tensor reading, we may want to make this a bit more
// composable/tensor agnostic at some point.
use crate::{Context, DType, Error as E, Layout, Result, Tensor};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::BufRead;

const VERBOSE: bool = false;

// DoS hardening for the `.pth` pickle VM (candle #3617). A crafted pickle can
// drive the value stack / memo to multi-GiB heap (`N`-floods, or `BINGET`
// replay of a memoised subtree) or build an arbitrarily deep value whose
// recursive `Drop` overflows the stack. These bound the VM's cumulative working
// set and its construction depth, mirroring `GGUF_MAX_VALUE_DEPTH` in
// `quantized/gguf_file.rs`. Availability-only: `reduce` never invokes callables.
//
// These are long-documented input-parsing weakness classes; see the MITRE CWE
// entries:
//   CWE-1325 (improperly controlled sequential memory allocation — the value
//   stack / memo-replay heap amplification): https://cwe.mitre.org/data/definitions/1325.html
//   CWE-674 (uncontrolled recursion — the deep-value `Drop` overflow):
//   https://cwe.mitre.org/data/definitions/674.html
//   CWE-770 (allocation of resources without limits — the per-item payload):
//   https://cwe.mitre.org/data/definitions/770.html
const PICKLE_MAX_WORKING_SET: u64 = 512 * 1024 * 1024;
const PICKLE_MAX_DEPTH: u32 = 64;
const PICKLE_MAX_PAYLOAD: u64 = 64 * 1024 * 1024;

// https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/
#[repr(u8)]
#[derive(Debug, Eq, PartialEq, Clone)]
pub enum OpCode {
    // https://github.com/python/cpython/blob/ed25f097160b5cbb0c9a1f9a746d2f1bbc96515a/Lib/pickletools.py#L2123
    Proto = 0x80,
    Global = b'c',
    BinPut = b'q',
    LongBinPut = b'r',
    EmptyTuple = b')',
    Reduce = b'R',
    Mark = b'(',
    BinUnicode = b'X',
    BinInt = b'J',
    Tuple = b't',
    BinPersId = b'Q',
    BinInt1 = b'K',
    BinInt2 = b'M',
    Tuple1 = 0x85,
    Tuple2 = 0x86,
    Tuple3 = 0x87,
    NewTrue = 0x88,
    NewFalse = 0x89,
    None = b'N',
    BinGet = b'h',
    LongBinGet = b'j',
    SetItem = b's',
    SetItems = b'u',
    EmptyDict = b'}',
    Dict = b'd',
    Build = b'b',
    Stop = b'.',
    NewObj = 0x81,
    EmptyList = b']',
    BinFloat = b'G',
    Append = b'a',
    Appends = b'e',
    Long1 = 0x8a,
}

// Avoid using FromPrimitive so as not to drag another dependency.
impl TryFrom<u8> for OpCode {
    type Error = u8;
    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        match value {
            0x80 => Ok(Self::Proto),
            b'c' => Ok(Self::Global),
            b'q' => Ok(Self::BinPut),
            b'r' => Ok(Self::LongBinPut),
            b')' => Ok(Self::EmptyTuple),
            b'R' => Ok(Self::Reduce),
            b'(' => Ok(Self::Mark),
            b'X' => Ok(Self::BinUnicode),
            b'J' => Ok(Self::BinInt),
            b't' => Ok(Self::Tuple),
            b'Q' => Ok(Self::BinPersId),
            b'K' => Ok(Self::BinInt1),
            b'M' => Ok(Self::BinInt2),
            b'N' => Ok(Self::None),
            0x85 => Ok(Self::Tuple1),
            0x86 => Ok(Self::Tuple2),
            0x87 => Ok(Self::Tuple3),
            0x88 => Ok(Self::NewTrue),
            0x89 => Ok(Self::NewFalse),
            b'h' => Ok(Self::BinGet),
            b'j' => Ok(Self::LongBinGet),
            b's' => Ok(Self::SetItem),
            b'u' => Ok(Self::SetItems),
            b'}' => Ok(Self::EmptyDict),
            b'd' => Ok(Self::EmptyDict),
            b'b' => Ok(Self::Build),
            b'.' => Ok(Self::Stop),
            0x81 => Ok(Self::NewObj),
            b']' => Ok(Self::EmptyList),
            b'G' => Ok(Self::BinFloat),
            b'a' => Ok(Self::Append),
            b'e' => Ok(Self::Appends),
            0x8a => Ok(Self::Long1),
            value => Err(value),
        }
    }
}

fn read_to_newline<R: BufRead>(r: &mut R) -> Result<Vec<u8>> {
    let mut data: Vec<u8> = Vec::with_capacity(32);
    r.read_until(b'\n', &mut data)?;
    data.pop();
    if data.last() == Some(&b'\r') {
        data.pop();
    }
    Ok(data)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Object {
    Class {
        module_name: String,
        class_name: String,
    },
    Int(i32),
    Long(i64),
    Float(f64),
    Unicode(String),
    Bool(bool),
    None,
    Tuple(Vec<Object>),
    List(Vec<Object>),
    Mark,
    Dict(Vec<(Object, Object)>),
    Reduce {
        callable: Box<Object>,
        args: Box<Object>,
    },
    Build {
        callable: Box<Object>,
        args: Box<Object>,
    },
    PersistentLoad(Box<Object>),
}

type OResult<T> = std::result::Result<T, Object>;

impl Object {
    pub fn unicode(self) -> OResult<String> {
        match self {
            Self::Unicode(t) => Ok(t),
            _ => Err(self),
        }
    }

    pub fn reduce(self) -> OResult<(Self, Self)> {
        match self {
            Self::Reduce { callable, args } => Ok((*callable, *args)),
            _ => Err(self),
        }
    }

    pub fn none(self) -> OResult<()> {
        match self {
            Self::None => Ok(()),
            _ => Err(self),
        }
    }

    pub fn persistent_load(self) -> OResult<Self> {
        match self {
            Self::PersistentLoad(t) => Ok(*t),
            _ => Err(self),
        }
    }

    pub fn bool(self) -> OResult<bool> {
        match self {
            Self::Bool(t) => Ok(t),
            _ => Err(self),
        }
    }

    pub fn int(self) -> OResult<i32> {
        match self {
            Self::Int(t) => Ok(t),
            _ => Err(self),
        }
    }

    pub fn int_or_long(self) -> OResult<i64> {
        match self {
            Self::Int(t) => Ok(t as i64),
            Self::Long(t) => Ok(t),
            _ => Err(self),
        }
    }

    pub fn tuple(self) -> OResult<Vec<Self>> {
        match self {
            Self::Tuple(t) => Ok(t),
            _ => Err(self),
        }
    }

    pub fn dict(self) -> OResult<Vec<(Self, Self)>> {
        match self {
            Self::Dict(t) => Ok(t),
            _ => Err(self),
        }
    }

    pub fn class(self) -> OResult<(String, String)> {
        match self {
            Self::Class {
                module_name,
                class_name,
            } => Ok((module_name, class_name)),
            _ => Err(self),
        }
    }

    pub fn into_tensor_info(
        self,
        name: Self,
        dir_name: &std::path::Path,
    ) -> Result<Option<TensorInfo>> {
        let name = match name.unicode() {
            Ok(name) => name,
            Err(_) => return Ok(None),
        };
        let (callable, args) = match self.reduce() {
            Ok(callable_args) => callable_args,
            _ => return Ok(None),
        };
        let (callable, args) = match callable {
            Object::Class {
                module_name,
                class_name,
            } if module_name == "torch._tensor" && class_name == "_rebuild_from_type_v2" => {
                let mut args = args.tuple()?;
                let callable = args.remove(0);
                let args = args.remove(1);
                (callable, args)
            }
            Object::Class {
                module_name,
                class_name,
            } if module_name == "torch._utils" && class_name == "_rebuild_parameter" => {
                let mut args = args.tuple()?;
                args.remove(0).reduce()?
            }
            _ => (callable, args),
        };
        match callable {
            Object::Class {
                module_name,
                class_name,
            } if module_name == "torch._utils" && class_name == "_rebuild_tensor_v2" => {}
            _ => return Ok(None),
        };
        let (layout, dtype, file_path, storage_size) = rebuild_args(args)?;
        Ok(Some(TensorInfo {
            name,
            dtype,
            layout,
            path: format!("{}/{}", dir_name.to_string_lossy(), file_path),
            storage_size,
        }))
    }
}

impl TryFrom<Object> for String {
    type Error = Object;
    fn try_from(value: Object) -> std::result::Result<Self, Self::Error> {
        match value {
            Object::Unicode(s) => Ok(s),
            other => Err(other),
        }
    }
}

impl TryFrom<Object> for usize {
    type Error = Object;
    fn try_from(value: Object) -> std::result::Result<Self, Self::Error> {
        match value {
            Object::Int(s) if s >= 0 => Ok(s as usize),
            other => Err(other),
        }
    }
}

impl<T: TryFrom<Object, Error = Object>> TryFrom<Object> for Vec<T> {
    type Error = Object;
    fn try_from(value: Object) -> std::result::Result<Self, Self::Error> {
        match value {
            Object::Tuple(values) => {
                // This does not return the appropriate value in the error case but instead return
                // the object related to the first error.
                values
                    .into_iter()
                    .map(|v| T::try_from(v))
                    .collect::<std::result::Result<Vec<T>, Self::Error>>()
            }
            other => Err(other),
        }
    }
}

#[derive(Debug)]
pub struct Stack {
    stack: Vec<Object>,
    // Structural nesting depth of each value in `stack`, kept length-synced so
    // depth is maintained in O(1) per opcode (a per-push deep walk would be an
    // O(n^2) CPU-DoS in the guard itself).
    depths: Vec<u32>,
    // Each memoised value carries its depth so `BINGET` restores it without
    // re-walking the cloned subtree.
    memo: HashMap<u32, (Object, u32)>,
    // Cumulative heap charged to the VM this parse; bounded by
    // `PICKLE_MAX_WORKING_SET`. Monotonic over-approximation (peak <= total).
    working_set: u64,
}

impl Stack {
    pub fn empty() -> Self {
        Self {
            stack: Vec::with_capacity(512),
            depths: Vec::with_capacity(512),
            memo: HashMap::new(),
            working_set: 0,
        }
    }

    // Charges `bytes` to the VM working set, rejecting once the permanent
    // `PICKLE_MAX_WORKING_SET` floor is crossed.
    fn charge(&mut self, bytes: u64) -> Result<()> {
        self.working_set = self.working_set.saturating_add(bytes);
        if self.working_set > PICKLE_MAX_WORKING_SET {
            crate::bail!(
                "pickle: working set {} exceeds limit {PICKLE_MAX_WORKING_SET}",
                self.working_set
            )
        }
        Ok(())
    }

    // Shallow heap of a single node: the enum slot plus its immediately-owned
    // string payload (children are charged when they are themselves pushed).
    fn shallow_size(obj: &Object) -> u64 {
        let base = std::mem::size_of::<Object>() as u64;
        let payload = match obj {
            Object::Unicode(s) => s.len() as u64,
            Object::Class {
                module_name,
                class_name,
            } => (module_name.len() + class_name.len()) as u64,
            _ => 0,
        };
        base.saturating_add(payload)
    }

    // Deep heap of a value and all its children. Recursive, but bounded to
    // `PICKLE_MAX_DEPTH` frames because no deeper value is ever constructed.
    // Used only on memo-clone opcodes, where a whole subtree is freshly cloned.
    fn deep_size(obj: &Object) -> u64 {
        let mut total = Self::shallow_size(obj);
        match obj {
            Object::Tuple(items) | Object::List(items) => {
                for it in items {
                    total = total.saturating_add(Self::deep_size(it));
                }
            }
            Object::Dict(pairs) => {
                for (k, v) in pairs {
                    total = total
                        .saturating_add(Self::deep_size(k))
                        .saturating_add(Self::deep_size(v));
                }
            }
            Object::PersistentLoad(inner) => {
                total = total.saturating_add(Self::deep_size(inner));
            }
            Object::Reduce { callable, args } | Object::Build { callable, args } => {
                total = total
                    .saturating_add(Self::deep_size(callable))
                    .saturating_add(Self::deep_size(args));
            }
            _ => {}
        }
        total
    }

    // Pushes a value at the given nesting depth — the only way `stack` grows.
    // Charges the value's shallow heap and rejects a depth past
    // `PICKLE_MAX_DEPTH`, bounding the flat-stack and nesting vectors here.
    fn push_at(&mut self, obj: Object, depth: u32) -> Result<()> {
        if depth > PICKLE_MAX_DEPTH {
            crate::bail!("pickle: value nesting depth {depth} exceeds limit {PICKLE_MAX_DEPTH}")
        }
        self.charge(Self::shallow_size(&obj))?;
        self.stack.push(obj);
        self.depths.push(depth);
        Ok(())
    }

    // Raises the top slot's depth after an in-place container mutation
    // (`APPEND(S)` / `SETITEM(S)` grow the top List/Dict without a fresh push).
    fn bump_top_depth(&mut self, child_depth: u32) -> Result<()> {
        let new = match self.depths.last() {
            Some(d) => (*d).max(child_depth.saturating_add(1)),
            None => return Ok(()),
        };
        if new > PICKLE_MAX_DEPTH {
            crate::bail!("pickle: value nesting depth {new} exceeds limit {PICKLE_MAX_DEPTH}")
        }
        if let Some(d) = self.depths.last_mut() {
            *d = new;
        }
        Ok(())
    }

    pub fn stack(&self) -> &[Object] {
        self.stack.as_slice()
    }

    pub fn read_loop<R: BufRead>(&mut self, r: &mut R) -> Result<()> {
        loop {
            if self.read(r)? {
                break;
            }
        }
        Ok(())
    }

    pub fn finalize(mut self) -> Result<Object> {
        self.pop()
    }

    // Pushes a leaf value (depth 1). Convenience over `push_at`.
    fn push(&mut self, obj: Object) -> Result<()> {
        self.push_at(obj, 1)
    }

    fn pop(&mut self) -> Result<Object> {
        match self.stack.pop() {
            None => crate::bail!("unexpected empty stack"),
            Some(obj) => {
                self.depths.pop();
                Ok(obj)
            }
        }
    }

    // Pops the top value together with its nesting depth, for opcodes that wrap
    // their operand one level deeper.
    fn pop_at(&mut self) -> Result<(Object, u32)> {
        match self.stack.pop() {
            None => crate::bail!("unexpected empty stack"),
            Some(obj) => {
                let depth = self.depths.pop().unwrap_or(1);
                Ok((obj, depth))
            }
        }
    }

    // https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/#Pickle.OpCodes.BUILD
    fn build(&mut self) -> Result<()> {
        let (args, da) = self.pop_at()?;
        let (obj, do_) = self.pop_at()?;
        let depth = da.max(do_).saturating_add(1);
        let obj = match (obj, args) {
            (Object::Dict(mut obj), Object::Dict(mut args)) => {
                obj.append(&mut args);
                Object::Dict(obj)
            }
            (obj, args) => Object::Build {
                callable: Box::new(obj),
                args: Box::new(args),
            },
        };
        self.push_at(obj, depth)
    }

    fn reduce(&mut self) -> Result<()> {
        let (args, da) = self.pop_at()?;
        let (callable, dc) = self.pop_at()?;
        #[allow(clippy::single_match)]
        let reduced = match &callable {
            Object::Class {
                module_name,
                class_name,
            } => {
                if module_name == "collections"
                    && (class_name == "OrderedDict" || class_name == "defaultdict")
                {
                    // TODO: have a separate ordered dict and a separate default dict.
                    Some(Object::Dict(vec![]))
                } else {
                    None
                }
            }
            _ => None,
        };
        match reduced {
            // A fresh empty dict — depth 1, independent of the operands.
            Some(empty) => self.push_at(empty, 1),
            None => self.push_at(
                Object::Reduce {
                    callable: Box::new(callable),
                    args: Box::new(args),
                },
                da.max(dc).saturating_add(1),
            ),
        }
    }

    fn last(&mut self) -> Result<&mut Object> {
        match self.stack.last_mut() {
            None => crate::bail!("unexpected empty stack"),
            Some(obj) => Ok(obj),
        }
    }

    // Clones a memoised value for `BINGET` replay, charging its deep heap to the
    // working set *before* the clone so an over-budget replay is rejected
    // without allocating the duplicate. Returns the value and its depth.
    fn memo_get(&mut self, id: u32) -> Result<(Object, u32)> {
        let (bytes, depth) = match self.memo.get(&id) {
            None => crate::bail!("missing object in memo {id}"),
            Some((obj, depth)) => (Self::deep_size(obj), *depth),
        };
        self.charge(bytes)?;
        match self.memo.get(&id) {
            None => crate::bail!("missing object in memo {id}"),
            Some((obj, _)) => Ok((obj.clone(), depth)),
        }
    }

    // Memoises the top value, charging its deep heap *before* the clone (the
    // `BINPUT` side of the replay-amplification bound).
    fn memo_put(&mut self, id: u32) -> Result<()> {
        let bytes = match self.stack.last() {
            Some(obj) => Self::deep_size(obj),
            None => crate::bail!("unexpected empty stack"),
        };
        self.charge(bytes)?;
        let obj = match self.stack.last() {
            Some(obj) => obj.clone(),
            None => crate::bail!("unexpected empty stack"),
        };
        let depth = self.depths.last().copied().unwrap_or(1);
        self.memo.insert(id, (obj, depth));
        Ok(())
    }

    fn persistent_load(&self, id: Object) -> Result<Object> {
        Ok(Object::PersistentLoad(Box::new(id)))
    }

    fn new_obj(&self, class: Object, args: Object) -> Result<Object> {
        Ok(Object::Reduce {
            callable: Box::new(class),
            args: Box::new(args),
        })
    }

    // Pops everything above the latest mark, returning the values and the
    // maximum nesting depth among them (0 if empty) so the caller can set the
    // container's depth in O(1).
    fn pop_to_marker(&mut self) -> Result<(Vec<Object>, u32)> {
        let mut mark_idx = None;
        for (idx, obj) in self.stack.iter().enumerate().rev() {
            if obj == &Object::Mark {
                mark_idx = Some(idx);
                break;
            }
        }
        match mark_idx {
            Some(mark_idx) => {
                let objs = self.stack.split_off(mark_idx + 1);
                let obj_depths = self.depths.split_off(mark_idx + 1);
                self.stack.pop();
                self.depths.pop();
                let max_depth = obj_depths.iter().copied().max().unwrap_or(0);
                Ok((objs, max_depth))
            }
            None => {
                crate::bail!("marker object not found")
            }
        }
    }

    pub fn read<R: BufRead>(&mut self, r: &mut R) -> Result<bool> {
        let op_code = match OpCode::try_from(r.read_u8()?) {
            Ok(op_code) => op_code,
            Err(op_code) => {
                crate::bail!("unknown op-code {op_code}")
            }
        };
        // println!("op: {op_code:?}");
        // println!("{:?}", self.stack);
        match op_code {
            OpCode::Proto => {
                let version = r.read_u8()?;
                if VERBOSE {
                    println!("proto {version}");
                }
            }
            OpCode::Global => {
                let module_name = read_to_newline(r)?;
                let class_name = read_to_newline(r)?;
                let module_name = String::from_utf8_lossy(&module_name).to_string();
                let class_name = String::from_utf8_lossy(&class_name).to_string();
                self.push(Object::Class {
                    module_name,
                    class_name,
                })?
            }
            OpCode::BinInt1 => {
                let arg = r.read_u8()?;
                self.push(Object::Int(arg as i32))?
            }
            OpCode::BinInt2 => {
                let arg = r.read_u16::<LittleEndian>()?;
                self.push(Object::Int(arg as i32))?
            }
            OpCode::BinInt => {
                let arg = r.read_i32::<LittleEndian>()?;
                self.push(Object::Int(arg))?
            }
            OpCode::BinFloat => {
                // Somehow floats are encoded using BigEndian whereas int types use LittleEndian.
                // https://github.com/python/cpython/blob/0c80da4c14d904a367968955544dd6ae58c8101c/Lib/pickletools.py#L855
                // https://github.com/pytorch/pytorch/blob/372d078f361e726bb4ac0884ac334b04c58179ef/torch/_weights_only_unpickler.py#L243
                let arg = r.read_f64::<byteorder::BigEndian>()?;
                self.push(Object::Float(arg))?
            }
            OpCode::BinUnicode => {
                let len = r.read_u32::<LittleEndian>()?;
                if len as u64 > PICKLE_MAX_PAYLOAD {
                    crate::bail!(
                        "pickle: BinUnicode length {len} exceeds limit {PICKLE_MAX_PAYLOAD}"
                    )
                }
                let mut data = vec![0u8; len as usize];
                r.read_exact(&mut data)?;
                let data = String::from_utf8(data).map_err(E::wrap)?;
                self.push(Object::Unicode(data))?
            }
            OpCode::BinPersId => {
                let (id, d) = self.pop_at()?;
                let obj = self.persistent_load(id)?;
                self.push_at(obj, d.saturating_add(1))?
            }
            OpCode::Tuple => {
                let (objs, md) = self.pop_to_marker()?;
                self.push_at(Object::Tuple(objs), md.saturating_add(1))?
            }
            OpCode::Tuple1 => {
                let (obj, d) = self.pop_at()?;
                self.push_at(Object::Tuple(vec![obj]), d.saturating_add(1))?
            }
            OpCode::Tuple2 => {
                let (obj2, d2) = self.pop_at()?;
                let (obj1, d1) = self.pop_at()?;
                self.push_at(
                    Object::Tuple(vec![obj1, obj2]),
                    d1.max(d2).saturating_add(1),
                )?
            }
            OpCode::Tuple3 => {
                let (obj3, d3) = self.pop_at()?;
                let (obj2, d2) = self.pop_at()?;
                let (obj1, d1) = self.pop_at()?;
                self.push_at(
                    Object::Tuple(vec![obj1, obj2, obj3]),
                    d1.max(d2).max(d3).saturating_add(1),
                )?
            }
            OpCode::NewTrue => self.push(Object::Bool(true))?,
            OpCode::NewFalse => self.push(Object::Bool(false))?,
            OpCode::Append => {
                let (value, dv) = self.pop_at()?;
                let pylist = self.last()?;
                if let Object::List(d) = pylist {
                    d.push(value)
                } else {
                    crate::bail!("expected a list, got {pylist:?}")
                }
                self.bump_top_depth(dv)?
            }
            OpCode::Appends => {
                let (objs, md) = self.pop_to_marker()?;
                let pylist = self.last()?;
                if let Object::List(d) = pylist {
                    d.extend(objs)
                } else {
                    crate::bail!("expected a list, got {pylist:?}")
                }
                self.bump_top_depth(md)?
            }
            OpCode::SetItem => {
                let (value, dv) = self.pop_at()?;
                let (key, dk) = self.pop_at()?;
                let pydict = self.last()?;
                if let Object::Dict(d) = pydict {
                    d.push((key, value))
                } else {
                    crate::bail!("expected a dict, got {pydict:?}")
                }
                self.bump_top_depth(dk.max(dv))?
            }
            OpCode::SetItems => {
                let (mut objs, md) = self.pop_to_marker()?;
                let pydict = self.last()?;
                if let Object::Dict(d) = pydict {
                    if objs.len() % 2 != 0 {
                        crate::bail!("setitems: not an even number of objects")
                    }
                    while let Some(value) = objs.pop() {
                        let key = objs.pop().context("empty objs")?;
                        d.push((key, value))
                    }
                } else {
                    crate::bail!("expected a dict, got {pydict:?}")
                }
                self.bump_top_depth(md)?
            }
            OpCode::None => self.push(Object::None)?,
            OpCode::Stop => {
                return Ok(true);
            }
            OpCode::Build => self.build()?,
            OpCode::EmptyDict => self.push(Object::Dict(vec![]))?,
            OpCode::Dict => {
                let (mut objs, md) = self.pop_to_marker()?;
                let mut pydict = vec![];
                if objs.len() % 2 != 0 {
                    crate::bail!("setitems: not an even number of objects")
                }
                while let Some(value) = objs.pop() {
                    let key = objs.pop().context("empty objs")?;
                    pydict.push((key, value))
                }
                self.push_at(Object::Dict(pydict), md.saturating_add(1))?
            }
            OpCode::Mark => self.push(Object::Mark)?,
            OpCode::Reduce => self.reduce()?,
            OpCode::EmptyTuple => self.push(Object::Tuple(vec![]))?,
            OpCode::EmptyList => self.push(Object::List(vec![]))?,
            OpCode::BinGet => {
                let arg = r.read_u8()?;
                let (obj, d) = self.memo_get(arg as u32)?;
                self.push_at(obj, d)?
            }
            OpCode::LongBinGet => {
                let arg = r.read_u32::<LittleEndian>()?;
                let (obj, d) = self.memo_get(arg)?;
                self.push_at(obj, d)?
            }
            OpCode::BinPut => {
                let arg = r.read_u8()?;
                self.memo_put(arg as u32)?
            }
            OpCode::LongBinPut => {
                let arg = r.read_u32::<LittleEndian>()?;
                self.memo_put(arg)?
            }
            OpCode::NewObj => {
                let (args, da) = self.pop_at()?;
                let (class, dc) = self.pop_at()?;
                let obj = self.new_obj(class, args)?;
                self.push_at(obj, da.max(dc).saturating_add(1))?
            }
            OpCode::Long1 => {
                let n_bytes = r.read_u8()?;
                // LONG1 is arbitrary-precision in Python; candle stores it as an
                // i64, so a value wider than 8 bytes is unrepresentable. Reject
                // it instead of panicking on the shift overflow a crafted
                // `n_bytes >= 9` triggered (`<< (i * 8)` with `i * 8 >= 64`).
                // CWE-190, integer overflow: https://cwe.mitre.org/data/definitions/190.html
                if n_bytes > 8 {
                    crate::bail!("pickle: LONG1 value too large ({n_bytes} bytes, max 8 for i64)")
                }
                let mut v = 0;
                // Decode the next n bytes in little endian
                for i in 0..n_bytes {
                    v |= (r.read_u8()? as i64) << (i * 8);
                }
                self.push(Object::Long(v))?
            }
        }
        Ok(false)
    }
}

impl From<Object> for E {
    fn from(value: Object) -> Self {
        E::Msg(format!("conversion error on {value:?}"))
    }
}

// https://github.com/pytorch/pytorch/blob/4eac43d046ded0f0a5a5fa8db03eb40f45bf656e/torch/_utils.py#L198
// Arguments: storage, storage_offset, size, stride, requires_grad, backward_hooks
fn rebuild_args(args: Object) -> Result<(Layout, DType, String, usize)> {
    let mut args = args.tuple()?;
    let stride = Vec::<usize>::try_from(args.remove(3))?;
    let size = Vec::<usize>::try_from(args.remove(2))?;
    let offset = args.remove(1).int_or_long()? as usize;
    let storage = args.remove(0).persistent_load()?;
    let mut storage = storage.tuple()?;
    let storage_size = storage.remove(4).int_or_long()? as usize;
    let path = storage.remove(2).unicode()?;
    let (_module_name, class_name) = storage.remove(1).class()?;
    let dtype = match class_name.as_str() {
        "FloatStorage" => DType::F32,
        "DoubleStorage" => DType::F64,
        "HalfStorage" => DType::F16,
        "BFloat16Storage" => DType::BF16,
        "ByteStorage" => DType::U8,
        "LongStorage" => DType::I64,
        other => {
            crate::bail!("unsupported storage type {other}")
        }
    };
    let layout = Layout::new(
        crate::Shape::from(size),
        stride,
        offset * dtype.size_in_bytes(),
    );
    Ok((layout, dtype, path, storage_size))
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub layout: Layout,
    pub path: String,
    pub storage_size: usize,
}

/// Read the tensor info from a .pth file.
///
/// # Arguments
/// * `file` - The path to the .pth file.
/// * `verbose` - Whether to print debug information.
/// * `key` - Optional key to retrieve `state_dict` from the pth file.
pub fn read_pth_tensor_info<P: AsRef<std::path::Path>>(
    file: P,
    verbose: bool,
    key: Option<&str>,
) -> Result<Vec<TensorInfo>> {
    let file = std::fs::File::open(file)?;
    let zip_reader = std::io::BufReader::new(file);
    let mut zip = zip::ZipArchive::new(zip_reader)?;
    let zip_file_names = zip
        .file_names()
        .map(|f| f.to_string())
        .collect::<Vec<String>>();

    let mut tensor_infos = vec![];
    for file_name in zip_file_names.iter() {
        if !file_name.ends_with("data.pkl") {
            continue;
        }
        let dir_name = std::path::PathBuf::from(file_name.strip_suffix(".pkl").context("no .pkl")?);
        let reader = zip.by_name(file_name)?;
        let mut reader = std::io::BufReader::new(reader);
        let mut stack = Stack::empty();
        stack.read_loop(&mut reader)?;
        let obj = stack.finalize()?;
        if VERBOSE || verbose {
            println!("{obj:#?}");
        }

        let obj = match obj {
            Object::Build { callable, args } => match *callable {
                Object::Reduce { callable, args: _ } => match *callable {
                    Object::Class {
                        module_name,
                        class_name,
                    } if module_name == "__torch__" && class_name == "Module" => *args,
                    _ => continue,
                },
                _ => continue,
            },
            obj => obj,
        };

        // If key is provided, then we need to extract the state_dict from the object.
        let obj = if let Some(key) = key {
            if let Object::Dict(key_values) = obj {
                key_values
                    .into_iter()
                    .find(|(k, _)| *k == Object::Unicode(key.to_owned()))
                    .map(|(_, v)| v)
                    .ok_or_else(|| E::Msg(format!("key {key} not found")))?
            } else {
                obj
            }
        } else {
            obj
        };

        // If the object is a dict, then we can extract the tensor info from it.
        // NOTE: We are assuming that the `obj` is state_dict by this stage.
        if let Object::Dict(key_values) = obj {
            for (name, value) in key_values.into_iter() {
                match value.into_tensor_info(name, &dir_name) {
                    Ok(Some(tensor_info)) => tensor_infos.push(tensor_info),
                    Ok(None) => {}
                    Err(err) => eprintln!("skipping: {err:?}"),
                }
            }
        }
    }
    Ok(tensor_infos)
}

/// Lazy tensor loader.
pub struct PthTensors {
    tensor_infos: HashMap<String, TensorInfo>,
    path: std::path::PathBuf,
    // We do not store a zip reader as it needs mutable access to extract data. Instead we
    // re-create a zip reader for each tensor.
}

impl PthTensors {
    pub fn new<P: AsRef<std::path::Path>>(path: P, key: Option<&str>) -> Result<Self> {
        let tensor_infos = read_pth_tensor_info(path.as_ref(), false, key)?;
        let tensor_infos = tensor_infos
            .into_iter()
            .map(|ti| (ti.name.to_string(), ti))
            .collect();
        let path = path.as_ref().to_owned();
        Ok(Self { tensor_infos, path })
    }

    pub fn tensor_infos(&self) -> &HashMap<String, TensorInfo> {
        &self.tensor_infos
    }

    pub fn get(&self, name: &str) -> Result<Option<Tensor>> {
        use std::io::Read;
        let tensor_info = match self.tensor_infos.get(name) {
            None => return Ok(None),
            Some(tensor_info) => tensor_info,
        };
        // We hope that the file has not changed since first reading it.
        let zip_reader = std::io::BufReader::new(std::fs::File::open(&self.path)?);
        let mut zip = zip::ZipArchive::new(zip_reader)?;
        let mut reader = zip.by_name(&tensor_info.path)?;
        let is_fortran_contiguous = tensor_info.layout.is_fortran_contiguous();
        let rank = tensor_info.layout.shape().rank();

        // Reading the data is a bit tricky as it can be strided, for now only support the basic
        // case and when the tensor is fortran contiguous.
        if !tensor_info.layout.is_contiguous() && !is_fortran_contiguous {
            crate::bail!(
                "cannot retrieve non-contiguous tensors {:?}",
                tensor_info.layout
            )
        }
        let start_offset = tensor_info.layout.start_offset();
        if start_offset > 0 {
            std::io::copy(
                &mut reader.by_ref().take(start_offset as u64),
                &mut std::io::sink(),
            )?;
        }
        let tensor = Tensor::from_reader(
            tensor_info.layout.shape().clone(),
            tensor_info.dtype,
            &mut reader,
        )?;

        if rank > 1 && is_fortran_contiguous {
            // Reverse the shape, e.g. Shape(2, 3, 4) -> Shape(4, 3, 2)
            let shape_reversed: Vec<_> = tensor_info.layout.dims().iter().rev().cloned().collect();
            let tensor = tensor.reshape(shape_reversed)?;

            // Permute (transpose) the dimensions, e.g. Shape(4, 3, 2) -> Shape(2, 3, 4)
            let dim_indices_reversed: Vec<_> = (0..rank).rev().collect();
            let tensor = tensor.permute(dim_indices_reversed)?;
            Ok(Some(tensor))
        } else {
            Ok(Some(tensor))
        }
    }
}

/// Read all the tensors from a PyTorch pth file with a given key.
///
/// # Arguments
/// * `path` - Path to the pth file.
/// * `key` - Optional key to retrieve `state_dict` from the pth file. Sometimes the pth file
///   contains multiple objects and the state_dict is the one we are interested in.
pub fn read_all_with_key<P: AsRef<std::path::Path>>(
    path: P,
    key: Option<&str>,
) -> Result<Vec<(String, Tensor)>> {
    let pth = PthTensors::new(path, key)?;
    let tensor_names = pth.tensor_infos.keys();
    let mut tensors = Vec::with_capacity(tensor_names.len());
    for name in tensor_names {
        if let Some(tensor) = pth.get(name)? {
            tensors.push((name.to_string(), tensor))
        }
    }
    Ok(tensors)
}

/// Read all the tensors from a PyTorch pth file.
///
/// # Arguments
/// * `path` - Path to the pth file.
pub fn read_all<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<(String, Tensor)>> {
    read_all_with_key(path, None)
}

#[cfg(test)]
mod dos_hardening_tests {
    // Regression tests for the pickle-VM DoS hardening (candle #3617). `Stack`
    // and `read_loop` are public, so these feed the VM directly.
    use super::*;

    // CWE-674: an unbounded `BINPERSID` (`Q`) chain wraps one `Box` deeper per
    // opcode, building an arbitrarily deep value whose recursive `Drop` would
    // overflow the stack. The depth cap rejects it after PICKLE_MAX_DEPTH levels
    // — and because *construction* is capped, the partial value left on the
    // stack is shallow and drops safely (so the test itself cannot overflow).
    #[test]
    fn rejects_deep_persid_chain() {
        let mut b = vec![0x80, 0x02, b'K', 0]; // PROTO 2; BININT1 0
        b.extend(std::iter::repeat(b'Q').take(1_000_000)); // 1e6 x BINPERSID
        b.push(b'.'); // STOP
        let mut s = Stack::empty();
        assert!(s.read_loop(&mut &b[..]).is_err());
    }

    // The per-item PICKLE_MAX_PAYLOAD cap rejects an oversized `BINUNICODE`
    // length *before* the backing buffer is allocated (no large allocation).
    #[test]
    fn rejects_oversized_unicode_pre_alloc() {
        let mut b = vec![0x80, 0x02, b'X']; // PROTO 2; BINUNICODE
        let too_big = (PICKLE_MAX_PAYLOAD as u32).saturating_add(1);
        b.extend(too_big.to_le_bytes());
        b.push(b'.');
        let mut s = Stack::empty();
        assert!(s.read_loop(&mut &b[..]).is_err());
    }

    // A small, well-formed pickle still parses unchanged.
    #[test]
    fn valid_pickle_still_parses() {
        // PROTO 2; BININT1 1/2/3; TUPLE3; STOP
        let b = vec![0x80, 0x02, b'K', 1, b'K', 2, b'K', 3, 0x87, b'.'];
        let mut s = Stack::empty();
        s.read_loop(&mut &b[..]).unwrap();
        let obj = s.finalize().unwrap();
        assert_eq!(
            obj,
            Object::Tuple(vec![Object::Int(1), Object::Int(2), Object::Int(3)])
        );
    }

    // candle #3617 (surfaced by fuzzing): a crafted LONG1 with `n_bytes >= 9`
    // used to panic with "attempt to shift left with overflow". It must now be
    // rejected cleanly instead of crashing.
    #[test]
    fn rejects_oversized_long1() {
        let mut b = vec![0x80, 0x02, 0x8a, 12]; // PROTO 2; LONG1; n_bytes = 12
        b.extend([0xFFu8; 12]);
        b.push(b'.'); // STOP
        let mut s = Stack::empty();
        assert!(s.read_loop(&mut &b[..]).is_err());
    }

    // CWE-1325: `BINGET` replay of a memoised list amplifies a few KB of opcodes
    // into multi-GB of heap; the working-set floor rejects it. Ignored by
    // default because it necessarily allocates up to the ~512 MiB cap before
    // bailing (run with `--ignored` to exercise it).
    #[test]
    #[ignore = "allocates up to the ~512 MiB working-set cap before rejecting"]
    fn rejects_memo_replay_amplification() {
        let mut b = vec![0x80, 0x02, b']', b'(']; // PROTO 2; EMPTY_LIST; MARK
        b.extend(std::iter::repeat([b'K', 1]).take(50_000).flatten()); // 50k x BININT1(1)
        b.push(b'e'); // APPENDS -> 50k-element list
        b.extend([b'q', 0]); // BINPUT 0 (memoise the list)
        b.extend(std::iter::repeat([b'h', 0]).take(20_000).flatten()); // BINGET 0 x20k
        b.push(b'.'); // STOP
        let mut s = Stack::empty();
        assert!(s.read_loop(&mut &b[..]).is_err());
    }
}
