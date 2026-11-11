use serde::Deserialize;
use serde_json::Value;
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

#[derive(Clone, Debug, Deserialize)]
pub struct SpirvGrammar {
    pub major_version: u16,
    pub minor_version: u16,
    pub revision: u16,
    pub instructions: Vec<SpirvInstruction>,
    pub operand_kinds: Vec<SpirvOperandKind>,
}

impl SpirvGrammar {
    pub fn new<P: AsRef<Path> + ?Sized>(path: &P) -> Self {
        let mut reader = BufReader::new(File::open(path).unwrap());
        let mut json = String::new();
        reader.read_to_string(&mut json).unwrap();
        serde_json::from_str(&json).unwrap()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpirvInstruction {
    pub opname: String,
    pub class: String,
    pub opcode: u16,
    #[serde(default)]
    pub operands: Vec<SpirvOperand>,
    // #[serde(default)]
    // pub capabilities: Vec<String>,
    // #[serde(default)]
    // pub extensions: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpirvOperand {
    pub kind: String,
    pub quantifier: Option<char>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpirvOperandKind {
    pub category: String,
    pub kind: String,
    #[serde(default)]
    pub enumerants: Vec<SpirvKindEnumerant>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpirvKindEnumerant {
    pub enumerant: String,
    pub value: Value,
    #[serde(default)]
    pub parameters: Vec<SpirvParameter>,
    // #[serde(default)]
    // pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpirvParameter {
    pub kind: String,
    pub name: Option<String>,
}
