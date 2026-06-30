//! Just enough pickle support to be able to read PyTorch checkpoints.
// This hardcodes objects that are required for tensor reading, we may want to make this a bit more
// composable/tensor agnostic at some point.
use crate::{Context, DType, Error as E, Layout, Result, Tensor};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::BufRead;

const VERBOSE: bool = false;

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
    /// Counts the total number of nodes in the object tree.
    /// Used for complexity accounting to prevent algorithmic-complexity DoS.
    ///
    /// Intentionally iterative (not recursive) so that a deeply-nested input
    /// cannot cause a native stack overflow before the complexity guard fires.
    fn node_count(&self) -> u64 {
        let mut count = 0u64;
        let mut work: Vec<&Object> = vec![self];
        while let Some(obj) = work.pop() {
            count += 1;
            match obj {
                Self::Tuple(v) | Self::List(v) => work.extend(v.iter()),
                Self::Dict(pairs) => {
                    for (k, v) in pairs {
                        work.push(k);
                        work.push(v);
                    }
                }
                Self::Reduce { callable, args } | Self::Build { callable, args } => {
                    work.push(callable);
                    work.push(args);
                }
                Self::PersistentLoad(inner) => work.push(inner),
                // Leaf nodes: Class, Int, Long, Float, Unicode, Bool, None, Mark
                _ => {}
            }
        }
        count
    }

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

/// Default cumulative node budget for unpickling. Prevents algorithmic-complexity DoS
/// from malicious pickles that exploit memo expansion to double object trees repeatedly.
const DEFAULT_MAX_COMPLEXITY: u64 = 1_000_000;

#[derive(Debug)]
pub struct Stack {
    stack: Vec<Object>,
    memo: HashMap<u32, Object>,
    /// Cumulative count of nodes cloned by memo retrievals so far.
    complexity: u64,
    /// Budget: an error is returned once `complexity` exceeds this value.
    max_complexity: u64,
}

impl Stack {
    pub fn empty() -> Self {
        Self::with_limit(DEFAULT_MAX_COMPLEXITY)
    }

    /// Create a stack with a custom complexity budget.
    ///
    /// The budget is consumed by `memo_get`: each retrieval charges the node
    /// count of the retrieved object. The counter accumulates across all calls
    /// to `read_loop` on the same `Stack` instance.
    ///
    /// A budget of `0` rejects any memo retrieval, including single leaf nodes.
    /// A budget of `u64::MAX` disables the limit check (node counting still
    /// runs on every retrieval, so there is a performance cost).
    pub fn with_limit(max_complexity: u64) -> Self {
        Self {
            stack: Vec::with_capacity(512),
            memo: HashMap::new(),
            complexity: 0,
            max_complexity,
        }
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

    fn push(&mut self, obj: Object) {
        self.stack.push(obj)
    }

    fn pop(&mut self) -> Result<Object> {
        match self.stack.pop() {
            None => crate::bail!("unexpected empty stack"),
            Some(obj) => Ok(obj),
        }
    }

    // https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/#Pickle.OpCodes.BUILD
    fn build(&mut self) -> Result<()> {
        let args = self.pop()?;
        let obj = self.pop()?;
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
        self.push(obj);
        Ok(())
    }

    fn reduce(&mut self) -> Result<()> {
        let args = self.pop()?;
        let callable = self.pop()?;
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
        let reduced = reduced.unwrap_or_else(|| Object::Reduce {
            callable: Box::new(callable),
            args: Box::new(args),
        });
        self.push(reduced);
        Ok(())
    }

    fn last(&mut self) -> Result<&mut Object> {
        match self.stack.last_mut() {
            None => crate::bail!("unexpected empty stack"),
            Some(obj) => Ok(obj),
        }
    }

    fn memo_get(&mut self, id: u32) -> Result<Object> {
        // Count nodes first (immutable borrow ends after this block) so we can
        // enforce the budget before performing the expensive clone.
        let cost = match self.memo.get(&id) {
            None => crate::bail!("missing object in memo {id}"),
            Some(obj) => obj.node_count(),
        };
        let new_complexity = self.complexity.saturating_add(cost);
        if new_complexity > self.max_complexity {
            crate::bail!(
                "pickle complexity limit exceeded ({} > {}); \
                 the input may be a malicious payload",
                new_complexity,
                self.max_complexity
            )
        }
        self.complexity = new_complexity;
        // Reborrow only after the budget check passes.
        self.memo
            .get(&id)
            .cloned()
            .ok_or_else(|| crate::Error::Msg(format!("missing object in memo {id}")))
    }

    fn memo_put(&mut self, id: u32) -> Result<()> {
        let obj = self.last()?.clone();
        self.memo.insert(id, obj);
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

    fn pop_to_marker(&mut self) -> Result<Vec<Object>> {
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
                self.stack.pop();
                Ok(objs)
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
                })
            }
            OpCode::BinInt1 => {
                let arg = r.read_u8()?;
                self.push(Object::Int(arg as i32))
            }
            OpCode::BinInt2 => {
                let arg = r.read_u16::<LittleEndian>()?;
                self.push(Object::Int(arg as i32))
            }
            OpCode::BinInt => {
                let arg = r.read_i32::<LittleEndian>()?;
                self.push(Object::Int(arg))
            }
            OpCode::BinFloat => {
                // Somehow floats are encoded using BigEndian whereas int types use LittleEndian.
                // https://github.com/python/cpython/blob/0c80da4c14d904a367968955544dd6ae58c8101c/Lib/pickletools.py#L855
                // https://github.com/pytorch/pytorch/blob/372d078f361e726bb4ac0884ac334b04c58179ef/torch/_weights_only_unpickler.py#L243
                let arg = r.read_f64::<byteorder::BigEndian>()?;
                self.push(Object::Float(arg))
            }
            OpCode::BinUnicode => {
                let len = r.read_u32::<LittleEndian>()?;
                let mut data = vec![0u8; len as usize];
                r.read_exact(&mut data)?;
                let data = String::from_utf8(data).map_err(E::wrap)?;
                self.push(Object::Unicode(data))
            }
            OpCode::BinPersId => {
                let id = self.pop()?;
                let obj = self.persistent_load(id)?;
                self.push(obj)
            }
            OpCode::Tuple => {
                let objs = self.pop_to_marker()?;
                self.push(Object::Tuple(objs))
            }
            OpCode::Tuple1 => {
                let obj = self.pop()?;
                self.push(Object::Tuple(vec![obj]))
            }
            OpCode::Tuple2 => {
                let obj2 = self.pop()?;
                let obj1 = self.pop()?;
                self.push(Object::Tuple(vec![obj1, obj2]))
            }
            OpCode::Tuple3 => {
                let obj3 = self.pop()?;
                let obj2 = self.pop()?;
                let obj1 = self.pop()?;
                self.push(Object::Tuple(vec![obj1, obj2, obj3]))
            }
            OpCode::NewTrue => self.push(Object::Bool(true)),
            OpCode::NewFalse => self.push(Object::Bool(false)),
            OpCode::Append => {
                let value = self.pop()?;
                let pylist = self.last()?;
                if let Object::List(d) = pylist {
                    d.push(value)
                } else {
                    crate::bail!("expected a list, got {pylist:?}")
                }
            }
            OpCode::Appends => {
                let objs = self.pop_to_marker()?;
                let pylist = self.last()?;
                if let Object::List(d) = pylist {
                    d.extend(objs)
                } else {
                    crate::bail!("expected a list, got {pylist:?}")
                }
            }
            OpCode::SetItem => {
                let value = self.pop()?;
                let key = self.pop()?;
                let pydict = self.last()?;
                if let Object::Dict(d) = pydict {
                    d.push((key, value))
                } else {
                    crate::bail!("expected a dict, got {pydict:?}")
                }
            }
            OpCode::SetItems => {
                let mut objs = self.pop_to_marker()?;
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
            }
            OpCode::None => self.push(Object::None),
            OpCode::Stop => {
                return Ok(true);
            }
            OpCode::Build => self.build()?,
            OpCode::EmptyDict => self.push(Object::Dict(vec![])),
            OpCode::Dict => {
                let mut objs = self.pop_to_marker()?;
                let mut pydict = vec![];
                if objs.len() % 2 != 0 {
                    crate::bail!("setitems: not an even number of objects")
                }
                while let Some(value) = objs.pop() {
                    let key = objs.pop().context("empty objs")?;
                    pydict.push((key, value))
                }
                self.push(Object::Dict(pydict))
            }
            OpCode::Mark => self.push(Object::Mark),
            OpCode::Reduce => self.reduce()?,
            OpCode::EmptyTuple => self.push(Object::Tuple(vec![])),
            OpCode::EmptyList => self.push(Object::List(vec![])),
            OpCode::BinGet => {
                let arg = r.read_u8()?;
                let obj = self.memo_get(arg as u32)?;
                self.push(obj)
            }
            OpCode::LongBinGet => {
                let arg = r.read_u32::<LittleEndian>()?;
                let obj = self.memo_get(arg)?;
                self.push(obj)
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
                let args = self.pop()?;
                let class = self.pop()?;
                let obj = self.new_obj(class, args)?;
                self.push(obj)
            }
            OpCode::Long1 => {
                let n_bytes = r.read_u8()?;
                let mut v = 0;
                // Decode the next n bytes in little endian
                for i in 0..n_bytes {
                    v |= (r.read_u8()? as i64) << (i * 8);
                }
                self.push(Object::Long(v))
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
mod tests {
    use super::*;
    use std::io::BufReader;

    /// Parse a raw pickle byte slice and return the top-level object.
    fn parse(bytes: &[u8]) -> crate::Result<Object> {
        let mut stack = Stack::empty();
        stack.read_loop(&mut BufReader::new(bytes))?;
        stack.finalize()
    }

    /// Same but with a custom complexity budget.
    fn parse_with_limit(bytes: &[u8], limit: u64) -> crate::Result<Object> {
        let mut stack = Stack::with_limit(limit);
        stack.read_loop(&mut BufReader::new(bytes))?;
        stack.finalize()
    }

    /// Build a pickle payload that stores an integer in memo slot 0 and then
    /// repeatedly retrieves it twice and wraps both copies in a Tuple2, storing
    /// the result back into the next memo slot. After `iters` doublings the
    /// memoised tree has 2^(iters+1)-1 nodes; the cumulative clone cost grows
    /// exponentially.
    fn make_tuple_doubling_payload(iters: u8) -> Vec<u8> {
        // OpCode bytes used below:
        //   Proto    = 0x80
        //   BinInt1  = 0x4b  ('K')
        //   BinPut   = 0x71  ('q')
        //   BinGet   = 0x68  ('h')
        //   Tuple2   = 0x86
        //   Stop     = 0x2e  ('.')
        // BinPut uses a u8 memo slot; i+1 would wrap at iters=255.
        assert!(iters < 255, "iters must be < 255 to avoid u8 memo-slot overflow");
        let mut p = vec![0x80u8, 0x02]; // Proto 2
        p.extend_from_slice(&[0x4b, 0x01, 0x71, 0x00]); // BinInt1(1), BinPut(0)
        for i in 0..iters {
            // BinGet(i), BinGet(i), Tuple2, BinPut(i+1)
            p.extend_from_slice(&[0x68, i, 0x68, i, 0x86, 0x71, i + 1]);
        }
        p.push(0x2e); // Stop
        p
    }

    /// Build a payload that stores a one-entry dict in memo slot 0 and then
    /// repeatedly retrieves it twice and merges them via BUILD, storing the
    /// result back. After `iters` doublings the dict has 2^iters entries.
    fn make_dict_doubling_payload(iters: u8) -> Vec<u8> {
        // Extra OpCode bytes:
        //   EmptyDict = 0x7d  ('}')
        //   SetItem   = 0x73  ('s')
        //   Build     = 0x62  ('b')
        // BinPut uses a u8 memo slot; i+1 would wrap at iters=255.
        assert!(iters < 255, "iters must be < 255 to avoid u8 memo-slot overflow");
        let mut p = vec![0x80u8, 0x02]; // Proto 2
        // EmptyDict, BinInt1(1), BinInt1(2), SetItem -> {1: 2}, BinPut(0)
        p.extend_from_slice(&[0x7d, 0x4b, 0x01, 0x4b, 0x02, 0x73, 0x71, 0x00]);
        for i in 0..iters {
            // BinGet(i), BinGet(i), Build, BinPut(i+1)
            p.extend_from_slice(&[0x68, i, 0x68, i, 0x62, 0x71, i + 1]);
        }
        p.push(0x2e); // Stop
        p
    }

    // --- normal-input regression: must continue to parse successfully ---

    #[test]
    fn test_normal_int() {
        // Proto 2, BinInt1(42), Stop  ->  Object::Int(42)
        let bytes = [0x80u8, 0x02, 0x4b, 0x2a, 0x2e];
        let obj = parse(&bytes).expect("should parse cleanly");
        assert_eq!(obj, Object::Int(42));
    }

    #[test]
    fn test_normal_memo_small() {
        // A few memo round-trips well within the default budget should succeed.
        let payload = make_tuple_doubling_payload(5); // 5 doublings: tiny
        parse(&payload).expect("small memo round-trips should succeed");
    }

    #[test]
    fn test_normal_dict_small() {
        let payload = make_dict_doubling_payload(5);
        parse(&payload).expect("small dict doublings should succeed");
    }

    /// Build a payload that creates a deeply-nested Tuple1 chain without memo
    /// doubling: each iteration retrieves memo[0], wraps it in Tuple1, and
    /// overwrites memo[0]. After `depth` iterations, memo[0] is a chain of
    /// `depth` nested Tuple1 wrappers. `node_count()` must walk this chain
    /// without overflowing the native call stack.
    fn make_deep_chain_payload(depth: u32) -> Vec<u8> {
        // OpCode bytes:
        //   Tuple1 = 0x85
        let mut p = vec![0x80u8, 0x02]; // Proto 2
        p.extend_from_slice(&[0x4b, 0x01, 0x71, 0x00]); // BinInt1(1), BinPut(0)
        for _ in 0..depth {
            // BinGet(0), Tuple1, BinPut(0)  — wraps and overwrites memo[0]
            p.extend_from_slice(&[0x68, 0x00, 0x85, 0x71, 0x00]);
        }
        p.push(0x2e); // Stop
        p
    }

    // --- complexity-limit regression: malicious payloads must be rejected ---

    #[test]
    fn test_tuple_doubling_exceeds_default_limit() {
        // 30 doublings produce a tree with ~2^31 nodes; cumulative clone cost
        // hits the 1 000 000-node budget long before that.
        let payload = make_tuple_doubling_payload(30);
        let err = parse(&payload).expect_err("should be rejected by complexity limit");
        assert!(
            err.to_string().contains("complexity"),
            "expected a complexity error, got: {err}"
        );
    }

    #[test]
    fn test_dict_build_doubling_exceeds_default_limit() {
        // Same idea but using dict merging via the BUILD opcode.
        let payload = make_dict_doubling_payload(30);
        let err = parse(&payload).expect_err("should be rejected by complexity limit");
        assert!(
            err.to_string().contains("complexity"),
            "expected a complexity error, got: {err}"
        );
    }

    #[test]
    fn test_deep_chain_does_not_stack_overflow() {
        // Payload containing 100_000 memo-expansion iterations. The complexity
        // budget fires at roughly iteration 1_414 (cumulative cost K*(K+1)/2
        // exceeds 1_000_000), so the parser never walks all 100_000 levels.
        // The payload is large to ensure a recursive node_count() would overflow
        // the native stack before the guard could fire; the iterative
        // implementation must reject cleanly without panicking.
        let payload = make_deep_chain_payload(100_000);
        // If parse panics (stack overflow), the test harness catches it and the
        // test fails — that is the "no stack overflow" guarantee. This assertion
        // only fires on a third case: parse returned an unexpected non-complexity error.
        let result = parse(&payload);
        assert!(
            result.is_ok() || result.unwrap_err().to_string().contains("complexity"),
            "parse returned an unexpected error type (expected ok or complexity limit)"
        );
    }

    #[test]
    fn test_custom_limit_respected() {
        // A single retrieval of a 3-node tuple (1 + 2 leaf ints) exceeds a
        // budget of 2 but fits in a budget of 4.
        let payload = make_tuple_doubling_payload(2); // memo[2] has 7 nodes; first retrieval costs 1, then 3
        // BinGet(0)×2 each cost 1 node (total 2, still within budget), then
        // BinGet(1) costs 3 nodes (total 5 > 2) — the error fires on the third
        // BinGet call, which is the first retrieval of the 3-node tuple.
        let err = parse_with_limit(&payload, 2)
            .expect_err("budget of 2 should be exceeded");
        assert!(err.to_string().contains("complexity"), "{err}");

        // Budget of 1_000_000 (default) should accept the 2-doubling payload.
        parse_with_limit(&payload, 1_000_000).expect("default budget should accept 2 doublings");
    }

    // --- PoC regression: exact payloads from the original CVE report ---

    #[test]
    fn test_poc_a_cpu_exhaustion() {
        // Reproduces PoC Path A from the original report:
        //
        //   NEWFALSE = b'\x89'; BINPUT = b'q'; BINGET = b'h'; TUPLE2 = b'\x86'; STOP = b'.'
        //   N = 20
        //   poc = NEWFALSE + BINPUT + b'\x00'
        //   for _ in range(N):
        //       poc += BINGET + b'\x00' + BINGET + b'\x00' + TUPLE2 + BINPUT + b'\x00'
        //   poc += STOP
        //
        // Before the fix, N=20 stalled for ~0.175 s and N=25 for ~5.7 s on a
        // release binary. After the fix the complexity limit fires during
        // iteration 18 (total charged nodes exceed 1_000_000) and returns
        // an error immediately.
        // No Proto header — matching the original PoC verbatim; Proto is optional.
        let mut poc = vec![
            0x89, // NewFalse  — push Bool(false)
            0x71, 0x00, // BinPut(0) — memo[0] = Bool(false)
        ];
        for _ in 0..20 {
            poc.extend_from_slice(&[
                0x68, 0x00, // BinGet(0)
                0x68, 0x00, // BinGet(0)
                0x86, // Tuple2
                0x71, 0x00, // BinPut(0) — overwrite memo[0] with doubled tree
            ]);
        }
        poc.push(0x2e); // Stop

        let err = parse(&poc).expect_err("PoC A must be rejected by complexity limit");
        assert!(
            err.to_string().contains("complexity"),
            "expected complexity error, got: {err}"
        );
    }

    #[test]
    fn test_poc_b_memory_exhaustion() {
        // Reproduces PoC Path B from the original report: the same exponential
        // doubling driven through BUILD dict-merge instead of Tuple2.
        //
        // Before the fix, ~92 GB RSS was measured on a 128 GB host before
        // termination. After the fix the complexity limit fires at the same
        // threshold as PoC A and returns an error immediately.
        let mut poc = vec![
            0x7d, // EmptyDict    — push {}
            0x4b, 0x01, // BinInt1(1) — key
            0x4b, 0x02, // BinInt1(2) — value
            0x73, // SetItem      — {1: 2}
            0x71, 0x00, // BinPut(0)   — memo[0] = {1: 2}
        ];
        for _ in 0..20 {
            poc.extend_from_slice(&[
                0x68, 0x00, // BinGet(0)
                0x68, 0x00, // BinGet(0)
                0x62, // Build  — dict-merge: doubles entry count
                0x71, 0x00, // BinPut(0) — overwrite memo[0]
            ]);
        }
        poc.push(0x2e); // Stop

        let err = parse(&poc).expect_err("PoC B must be rejected by complexity limit");
        assert!(
            err.to_string().contains("complexity"),
            "expected complexity error, got: {err}"
        );
    }

    #[test]
    fn test_unlimited_budget_allows_large_payload() {
        // with_limit(u64::MAX) is documented to disable the limit check.
        // 20 iterations exceed the default 1 000 000-node budget (which fires at
        // iteration 17) but produce a final tree of only ~2 M nodes (~100 MB),
        // safe to materialise on any development machine.
        // Previously used 30 iterations (2^31 nodes, ~50 GB), which crashed the host.
        let payload = make_tuple_doubling_payload(20);
        parse_with_limit(&payload, u64::MAX).expect("unlimited budget should allow any payload");
    }

    #[test]
    fn test_zero_limit_rejects_any_memo_get() {
        // Budget of 0 is documented to reject any memo retrieval, even a single
        // leaf node. The very first BinGet charges 1 node > 0, so it must error.
        let payload = make_tuple_doubling_payload(1); // one BinGet of Int(1) before doubling
        let err = parse_with_limit(&payload, 0).expect_err("budget of 0 should reject memo gets");
        assert!(err.to_string().contains("complexity"), "{err}");
    }

    #[test]
    fn test_zero_limit_allows_pickle_with_no_memo_get() {
        // A pickle that never calls BinGet charges nothing, so a budget of 0
        // must not prevent it from parsing.
        // Proto 2, BinInt1(42), Stop — no memo access at all.
        let bytes = [0x80u8, 0x02, 0x4b, 0x2a, 0x2e];
        let obj = parse_with_limit(&bytes, 0).expect("no-memo pickle should succeed with budget 0");
        assert_eq!(obj, Object::Int(42));
    }
}
