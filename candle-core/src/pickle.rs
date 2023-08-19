// Just enough pickle support to be able to read PyTorch checkpoints.
// This hardcodes objects that are required for tensor reading, we may want to make this a bit more
// composable/tensor agnostic at some point.
use crate::{DType, Error as E, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::BufRead;

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
    Unicode(String),
    Bool(bool),
    None,
    Tuple(Vec<Object>),
    Mark,
    Dict(Vec<(Object, Object)>),
    Reduce {
        callable: Box<Object>,
        args: Box<Object>,
    },
    PersistentLoad(Box<Object>),
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
    memo: HashMap<u32, Object>,
}

impl Stack {
    pub fn empty() -> Self {
        Self {
            stack: Vec::with_capacity(512),
            memo: HashMap::new(),
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
        let mut args = self.pop()?;
        let obj = self.last()?;
        match (obj, &mut args) {
            (Object::Dict(obj), Object::Dict(args)) => obj.append(args),
            (obj, args) => println!("build {obj:?} {args:?}"),
        }
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
                if module_name == "collections" && class_name == "OrderedDict" {
                    // TODO: have a separate ordered dict.
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

    fn memo_get(&self, id: u32) -> Result<Object> {
        match self.memo.get(&id) {
            None => crate::bail!("missing object in memo {id}"),
            Some(obj) => {
                // Maybe we should use refcounting rather than doing potential large clones here.
                Ok(obj.clone())
            }
        }
    }

    fn memo_put(&mut self, id: u32) -> Result<()> {
        let obj = self.last()?.clone();
        self.memo.insert(id, obj);
        Ok(())
    }

    fn persistent_load(&self, id: Object) -> Result<Object> {
        Ok(Object::PersistentLoad(Box::new(id)))
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
                println!("proto {version}");
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
                        let key = objs.pop().unwrap();
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
                    let key = objs.pop().unwrap();
                    pydict.push((key, value))
                }
                self.push(Object::Dict(pydict))
            }
            OpCode::Mark => self.push(Object::Mark),
            OpCode::Reduce => self.reduce()?,
            OpCode::EmptyTuple => self.push(Object::Tuple(vec![])),
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
        }
        Ok(false)
    }
}

pub fn read_pth_tensor_info<P: AsRef<std::path::Path>>(
    file: P,
) -> Result<Vec<(String, DType, Vec<usize>)>> {
    let file = std::fs::File::open(file)?;
    let zip_reader = std::io::BufReader::new(file);
    let mut zip = zip::ZipArchive::new(zip_reader)?;
    let zip_file_names = zip
        .file_names()
        .map(|f| f.to_string())
        .collect::<Vec<String>>();

    let mut tensor_info = vec![];
    for name in zip_file_names.iter() {
        if !name.ends_with("data.pkl") {
            continue;
        }
        let reader = zip.by_name(name)?;
        let mut reader = std::io::BufReader::new(reader);
        let mut stack = Stack::empty();
        stack.read_loop(&mut reader)?;
        let obj = stack.finalize()?;
        if let Object::Dict(key_values) = obj {
            for (key, value) in key_values.into_iter() {
                let key = match String::try_from(key) {
                    Ok(key) => key,
                    Err(_) => continue,
                };
                let (callable, args) = match value {
                    Object::Reduce { callable, args } => (*callable, *args),
                    _ => continue,
                };
                match callable {
                    Object::Class {
                        module_name,
                        class_name,
                    } if module_name == "torch._utils" && class_name == "_rebuild_tensor_v2" => {}
                    _ => continue,
                };
                // https://github.com/pytorch/pytorch/blob/4eac43d046ded0f0a5a5fa8db03eb40f45bf656e/torch/_utils.py#L198
                // Arguments: storage, storage_offset, size, stride, requires_grad, backward_hooks
                let mut args = match args {
                    Object::Tuple(args) => args,
                    _ => continue,
                };
                let size = match Vec::<usize>::try_from(args.remove(2)) {
                    Ok(size) => size,
                    Err(_) => continue,
                };
                let storage = match args.remove(0) {
                    Object::PersistentLoad(vs) => *vs,
                    _ => continue,
                };
                let storage = match storage {
                    Object::Tuple(vs) => vs,
                    _ => continue,
                };
                let dtype = match &storage[1] {
                    Object::Class { class_name, .. } => match class_name.as_str() {
                        "FloatStorage" => DType::F32,
                        "DoubleStorage" => DType::F64,
                        "HalfStorage" => DType::F16,
                        "BFloat16Storage" => DType::BF16,
                        "ByteStorage" => DType::U8,
                        other => {
                            eprintln!("unsupported storage type {other}");
                            continue;
                        }
                    },
                    _ => continue,
                };
                tensor_info.push((key, dtype, size))
                // pass
            }
        }
    }
    Ok(tensor_info)
}
