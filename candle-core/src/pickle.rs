use crate::{Error as E, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::BufRead;

// https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/
#[repr(u8)]
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

#[derive(Debug)]
enum Object {
    Class {
        module_name: String,
        class_name: String,
    },
    Int(i32),
    Unicode(String),
}

#[derive(Debug)]
pub struct Stack {
    stack: Vec<Object>,
}

impl Stack {
    pub fn empty() -> Self {
        Self {
            stack: Vec::with_capacity(512),
        }
    }

    pub fn read_loop<R: BufRead>(&mut self, r: &mut R) -> Result<()> {
        loop {
            if self.read(r)? {
                break;
            }
        }
        Ok(())
    }

    pub fn read<R: BufRead>(&mut self, r: &mut R) -> Result<bool> {
        let op_code = match OpCode::try_from(r.read_u8()?) {
            Ok(op_code) => op_code,
            Err(op_code) => {
                crate::bail!("unknown op-code {op_code}")
            }
        };
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
                self.stack.push(Object::Class {
                    module_name,
                    class_name,
                })
            }
            OpCode::BinInt1 => {
                let arg = r.read_u8()?;
                self.stack.push(Object::Int(arg as i32))
            }
            OpCode::BinInt2 => {
                let arg = r.read_u16::<LittleEndian>()?;
                self.stack.push(Object::Int(arg as i32))
            }
            OpCode::BinInt => {
                let arg = r.read_i32::<LittleEndian>()?;
                self.stack.push(Object::Int(arg))
            }
            OpCode::BinUnicode => {
                let len = r.read_u32::<LittleEndian>()?;
                let mut data = vec![0u8; len as usize];
                r.read_exact(&mut data)?;
                let data = String::from_utf8(data).map_err(E::wrap)?;
                self.stack.push(Object::Unicode(data))
            }
            OpCode::BinPersId => {
                println!("binpersid");
            }
            OpCode::Tuple => {
                println!("tuple");
            }
            OpCode::Tuple1 => {
                println!("tuple1");
            }
            OpCode::Tuple2 => {
                println!("tuple2");
            }
            OpCode::Tuple3 => {
                println!("tuple3");
            }
            OpCode::NewTrue => {
                println!("true");
            }
            OpCode::NewFalse => {
                println!("false");
            }
            OpCode::SetItem => {
                println!("setitem");
            }
            OpCode::SetItems => {
                println!("setitems");
            }
            OpCode::None => {
                println!("none");
            }
            OpCode::Stop => {
                println!("stop");
                return Ok(true);
            }
            OpCode::Build => {
                println!("build");
            }
            OpCode::EmptyDict => {
                println!("emptydict");
            }
            OpCode::Dict => {
                println!("dict");
            }
            OpCode::Mark => {
                println!("mark");
            }
            OpCode::Reduce => {
                println!("reduce");
            }
            OpCode::EmptyTuple => {
                println!("empty-tuple");
            }
            OpCode::BinGet => {
                let arg = r.read_u8()?;
                println!("binget {arg}");
            }
            OpCode::LongBinGet => {
                let arg = r.read_u32::<LittleEndian>()?;
                println!("binget {arg}");
            }
            OpCode::BinPut => {
                let arg = r.read_u8()?;
                println!("binput {arg}");
            }
            OpCode::LongBinPut => {
                let arg = r.read_u32::<LittleEndian>()?;
                println!("binput {arg}");
            }
        }
        Ok(false)
    }
}
