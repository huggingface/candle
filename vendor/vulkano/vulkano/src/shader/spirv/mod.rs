//! Parsing and analysis utilities for SPIR-V shader binaries.
//!
//! This can be used to inspect and validate a SPIR-V module at runtime. The `Spirv` type does some
//! validation, but you should not assume that code that is read successfully is valid.
//!
//! For more information about SPIR-V modules, instructions and types, see the
//! [SPIR-V specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html).

use crate::{shader::SpecializationConstant, Version};
use foldhash::{HashMap, HashSet};
use smallvec::{smallvec, SmallVec};
use std::{
    borrow::Cow,
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    string::FromUtf8Error,
};

mod specialization;

include!(crate::autogen_output!("spirv_parse.rs"));

/// A parsed and analyzed SPIR-V module.
#[derive(Clone, Debug)]
pub struct Spirv {
    version: Version,
    bound: u32,
    ids: HashMap<Id, IdInfo>,

    // Items described in the spec section "Logical Layout of a Module"
    capabilities: Vec<Instruction>,
    extensions: Vec<Instruction>,
    ext_inst_imports: Vec<Instruction>,
    memory_model: Instruction,
    entry_points: Vec<Instruction>,
    execution_modes: Vec<Instruction>,
    names: Vec<Instruction>,
    decorations: Vec<Instruction>,
    types: Vec<Instruction>,
    constants: Vec<Instruction>,
    global_variables: Vec<Instruction>,
    functions: HashMap<Id, FunctionInfo>,
}

impl Spirv {
    /// Parses a SPIR-V document from a list of words.
    pub fn new(words: &[u32]) -> Result<Spirv, SpirvError> {
        if words.len() < 5 {
            return Err(SpirvError::InvalidHeader);
        }

        if words[0] != 0x07230203 {
            return Err(SpirvError::InvalidHeader);
        }

        let version = Version {
            major: (words[1] & 0x00ff0000) >> 16,
            minor: (words[1] & 0x0000ff00) >> 8,
            patch: words[1] & 0x000000ff,
        };

        // For safety, we recalculate the bound ourselves.
        let mut bound = 0;
        let mut ids = HashMap::default();

        let mut capabilities = Vec::new();
        let mut extensions = Vec::new();
        let mut ext_inst_imports = Vec::new();
        let mut memory_models = Vec::new();
        let mut entry_points = Vec::new();
        let mut execution_modes = Vec::new();
        let mut names = Vec::new();
        let mut decorations = Vec::new();
        let mut types = Vec::new();
        let mut constants = Vec::new();
        let mut global_variables = Vec::new();

        let mut functions = HashMap::default();
        let mut current_function: Option<&mut FunctionInfo> = None;

        for instruction in iter_instructions(&words[5..]) {
            let instruction = instruction?;

            if let Some(id) = instruction.result_id() {
                bound = bound.max(u32::from(id) + 1);

                let members = if let Instruction::TypeStruct {
                    ref member_types, ..
                } = instruction
                {
                    member_types
                        .iter()
                        .map(|_| StructMemberInfo::default())
                        .collect()
                } else {
                    Vec::new()
                };

                let data = IdInfo {
                    instruction: instruction.clone(),
                    names: Vec::new(),
                    decorations: Vec::new(),
                    members,
                };

                if ids.insert(id, data).is_some() {
                    return Err(SpirvError::DuplicateId { id });
                }
            }

            if matches!(instruction, Instruction::Line { .. } | Instruction::NoLine) {
                continue;
            }

            match instruction {
                Instruction::Function { result_id, .. } => {
                    current_function = None;
                    let function = functions.entry(result_id).or_insert_with(|| {
                        let entry_point = entry_points
                            .iter()
                            .find(|instruction| {
                                matches!(
                                    **instruction,
                                    Instruction::EntryPoint { entry_point, .. }
                                    if entry_point == result_id
                                )
                            })
                            .cloned();
                        let execution_modes = execution_modes
                            .iter()
                            .filter(|instruction| {
                                matches!(
                                    **instruction,
                                    Instruction::ExecutionMode { entry_point, .. }
                                    | Instruction::ExecutionModeId { entry_point, .. }
                                    if entry_point == result_id
                                )
                            })
                            .cloned()
                            .collect();

                        FunctionInfo {
                            instructions: Vec::new(),
                            called_functions: HashSet::default(),
                            entry_point,
                            execution_modes,
                        }
                    });
                    let current_function = current_function.insert(function);
                    current_function.instructions.push(instruction);
                }
                Instruction::FunctionEnd => {
                    let current_function = current_function.take().unwrap();
                    current_function.instructions.push(instruction);
                }
                _ => {
                    if let Some(current_function) = current_function.as_mut() {
                        if let Instruction::FunctionCall { function, .. } = instruction {
                            current_function.called_functions.insert(function);
                        }

                        current_function.instructions.push(instruction);
                    } else {
                        let destination = match instruction {
                            Instruction::Capability { .. } => &mut capabilities,
                            Instruction::Extension { .. } => &mut extensions,
                            Instruction::ExtInstImport { .. } => &mut ext_inst_imports,
                            Instruction::MemoryModel { .. } => &mut memory_models,
                            Instruction::EntryPoint { .. } => &mut entry_points,
                            Instruction::ExecutionMode { .. }
                            | Instruction::ExecutionModeId { .. } => &mut execution_modes,
                            Instruction::Name { .. } | Instruction::MemberName { .. } => &mut names,
                            Instruction::Decorate { .. }
                            | Instruction::MemberDecorate { .. }
                            | Instruction::DecorationGroup { .. }
                            | Instruction::GroupDecorate { .. }
                            | Instruction::GroupMemberDecorate { .. }
                            | Instruction::DecorateId { .. }
                            | Instruction::DecorateString { .. }
                            | Instruction::MemberDecorateString { .. } => &mut decorations,
                            Instruction::TypeVoid { .. }
                            | Instruction::TypeBool { .. }
                            | Instruction::TypeInt { .. }
                            | Instruction::TypeFloat { .. }
                            | Instruction::TypeVector { .. }
                            | Instruction::TypeMatrix { .. }
                            | Instruction::TypeImage { .. }
                            | Instruction::TypeSampler { .. }
                            | Instruction::TypeSampledImage { .. }
                            | Instruction::TypeArray { .. }
                            | Instruction::TypeRuntimeArray { .. }
                            | Instruction::TypeStruct { .. }
                            | Instruction::TypeOpaque { .. }
                            | Instruction::TypePointer { .. }
                            | Instruction::TypeFunction { .. }
                            | Instruction::TypeEvent { .. }
                            | Instruction::TypeDeviceEvent { .. }
                            | Instruction::TypeReserveId { .. }
                            | Instruction::TypeQueue { .. }
                            | Instruction::TypePipe { .. }
                            | Instruction::TypeForwardPointer { .. }
                            | Instruction::TypePipeStorage { .. }
                            | Instruction::TypeNamedBarrier { .. }
                            | Instruction::TypeRayQueryKHR { .. }
                            | Instruction::TypeAccelerationStructureKHR { .. }
                            | Instruction::TypeCooperativeMatrixNV { .. }
                            | Instruction::TypeVmeImageINTEL { .. }
                            | Instruction::TypeAvcImePayloadINTEL { .. }
                            | Instruction::TypeAvcRefPayloadINTEL { .. }
                            | Instruction::TypeAvcSicPayloadINTEL { .. }
                            | Instruction::TypeAvcMcePayloadINTEL { .. }
                            | Instruction::TypeAvcMceResultINTEL { .. }
                            | Instruction::TypeAvcImeResultINTEL { .. }
                            | Instruction::TypeAvcImeResultSingleReferenceStreamoutINTEL {
                                ..
                            }
                            | Instruction::TypeAvcImeResultDualReferenceStreamoutINTEL { .. }
                            | Instruction::TypeAvcImeSingleReferenceStreaminINTEL { .. }
                            | Instruction::TypeAvcImeDualReferenceStreaminINTEL { .. }
                            | Instruction::TypeAvcRefResultINTEL { .. }
                            | Instruction::TypeAvcSicResultINTEL { .. } => &mut types,
                            Instruction::ConstantTrue { .. }
                            | Instruction::ConstantFalse { .. }
                            | Instruction::Constant { .. }
                            | Instruction::ConstantComposite { .. }
                            | Instruction::ConstantSampler { .. }
                            | Instruction::ConstantNull { .. }
                            | Instruction::ConstantPipeStorage { .. }
                            | Instruction::SpecConstantTrue { .. }
                            | Instruction::SpecConstantFalse { .. }
                            | Instruction::SpecConstant { .. }
                            | Instruction::SpecConstantComposite { .. }
                            | Instruction::SpecConstantOp { .. }
                            | Instruction::Undef { .. } => &mut constants,
                            Instruction::Variable { .. } => &mut global_variables,
                            _ => continue,
                        };

                        destination.push(instruction);
                    }
                }
            }
        }

        let memory_model = memory_models.drain(..).next().unwrap();

        // Add decorations to ids,
        // while also expanding decoration groups into individual decorations.
        let mut decoration_groups: HashMap<Id, Vec<Instruction>> = HashMap::default();
        let decorations = decorations
            .into_iter()
            .flat_map(|instruction| -> SmallVec<[Instruction; 1]> {
                match instruction {
                    Instruction::Decorate { target, .. }
                    | Instruction::DecorateId { target, .. }
                    | Instruction::DecorateString { target, .. } => {
                        let id_info = ids.get_mut(&target).unwrap();

                        if matches!(id_info.instruction(), Instruction::DecorationGroup { .. }) {
                            decoration_groups
                                .entry(target)
                                .or_default()
                                .push(instruction);
                            smallvec![]
                        } else {
                            id_info.decorations.push(instruction.clone());
                            smallvec![instruction]
                        }
                    }
                    Instruction::MemberDecorate {
                        structure_type: target,
                        member,
                        ..
                    }
                    | Instruction::MemberDecorateString {
                        struct_type: target,
                        member,
                        ..
                    } => {
                        ids.get_mut(&target).unwrap().members[member as usize]
                            .decorations
                            .push(instruction.clone());
                        smallvec![instruction]
                    }
                    Instruction::DecorationGroup { result_id } => {
                        // Drop the instruction altogether.
                        decoration_groups.entry(result_id).or_default();
                        ids.remove(&result_id);
                        smallvec![]
                    }
                    Instruction::GroupDecorate {
                        decoration_group,
                        ref targets,
                    } => {
                        let decorations = &decoration_groups[&decoration_group];

                        targets
                            .iter()
                            .copied()
                            .flat_map(|target| {
                                decorations
                                    .iter()
                                    .map(move |instruction| (target, instruction))
                            })
                            .map(|(target, instruction)| {
                                let id_info = ids.get_mut(&target).unwrap();

                                match instruction {
                                    Instruction::Decorate { decoration, .. } => {
                                        let instruction = Instruction::Decorate {
                                            target,
                                            decoration: decoration.clone(),
                                        };
                                        id_info.decorations.push(instruction.clone());
                                        instruction
                                    }
                                    Instruction::DecorateId { decoration, .. } => {
                                        let instruction = Instruction::DecorateId {
                                            target,
                                            decoration: decoration.clone(),
                                        };
                                        id_info.decorations.push(instruction.clone());
                                        instruction
                                    }
                                    _ => unreachable!(),
                                }
                            })
                            .collect()
                    }
                    Instruction::GroupMemberDecorate {
                        decoration_group,
                        ref targets,
                    } => {
                        let decorations = &decoration_groups[&decoration_group];

                        targets
                            .iter()
                            .copied()
                            .flat_map(|target| {
                                decorations
                                    .iter()
                                    .map(move |instruction| (target, instruction))
                            })
                            .map(|((structure_type, member), instruction)| {
                                let member_info =
                                    &mut ids.get_mut(&structure_type).unwrap().members
                                        [member as usize];

                                match instruction {
                                    Instruction::Decorate { decoration, .. } => {
                                        let instruction = Instruction::MemberDecorate {
                                            structure_type,
                                            member,
                                            decoration: decoration.clone(),
                                        };
                                        member_info.decorations.push(instruction.clone());
                                        instruction
                                    }
                                    Instruction::DecorateId { .. } => {
                                        panic!(
                                            "a DecorateId instruction targets a decoration group, \
                                            and that decoration group is applied using a \
                                            GroupMemberDecorate instruction, but there is no \
                                            MemberDecorateId instruction"
                                        );
                                    }
                                    _ => unreachable!(),
                                }
                            })
                            .collect()
                    }
                    _ => smallvec![instruction],
                }
            })
            .collect();

        names.retain(|instruction| match *instruction {
            Instruction::Name { target, .. } => {
                if let Some(id_info) = ids.get_mut(&target) {
                    id_info.names.push(instruction.clone());
                    true
                } else {
                    false
                }
            }
            Instruction::MemberName { ty, member, .. } => {
                if let Some(id_info) = ids.get_mut(&ty) {
                    id_info.members[member as usize]
                        .names
                        .push(instruction.clone());
                    true
                } else {
                    false
                }
            }
            _ => unreachable!(),
        });

        Ok(Spirv {
            version,
            bound,
            ids,

            capabilities,
            extensions,
            ext_inst_imports,
            memory_model,
            entry_points,
            execution_modes,
            names,
            decorations,
            types,
            constants,
            global_variables,
            functions,
        })
    }

    /// Returns the SPIR-V version that the module is compiled for.
    #[inline]
    pub fn version(&self) -> Version {
        self.version
    }

    /// Returns information about an `Id`.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not defined in this module. This can in theory only happen if you are
    ///   mixing `Id`s from different modules.
    #[inline]
    pub fn id(&self, id: Id) -> &IdInfo {
        &self.ids[&id]
    }

    /// Returns the function with the given `id`, if it exists.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not defined in this module. This can in theory only happen if you are
    ///   mixing `Id`s from different modules.
    #[inline]
    pub fn function(&self, id: Id) -> &FunctionInfo {
        &self.functions[&id]
    }

    /// Returns all `Capability` instructions.
    #[inline]
    pub fn capabilities(&self) -> &[Instruction] {
        &self.capabilities
    }

    /// Returns all `Extension` instructions.
    #[inline]
    pub fn extensions(&self) -> &[Instruction] {
        &self.extensions
    }

    /// Returns all `ExtInstImport` instructions.
    #[inline]
    pub fn ext_inst_imports(&self) -> &[Instruction] {
        &self.ext_inst_imports
    }

    /// Returns the `MemoryModel` instruction.
    #[inline]
    pub fn memory_model(&self) -> &Instruction {
        &self.memory_model
    }

    /// Returns all `EntryPoint` instructions.
    #[inline]
    pub fn entry_points(&self) -> &[Instruction] {
        &self.entry_points
    }

    /// Returns all execution mode instructions.
    #[inline]
    pub fn execution_modes(&self) -> &[Instruction] {
        &self.execution_modes
    }

    /// Returns all name debug instructions.
    #[inline]
    pub fn names(&self) -> &[Instruction] {
        &self.names
    }

    /// Returns all decoration instructions.
    #[inline]
    pub fn decorations(&self) -> &[Instruction] {
        &self.decorations
    }

    /// Returns all type instructions.
    #[inline]
    pub fn types(&self) -> &[Instruction] {
        &self.types
    }

    /// Returns all constant and specialization constant instructions.
    #[inline]
    pub fn constants(&self) -> &[Instruction] {
        &self.constants
    }

    /// Returns all global variable instructions.
    #[inline]
    pub fn global_variables(&self) -> &[Instruction] {
        &self.global_variables
    }

    /// Returns all functions.
    #[inline]
    pub fn functions(&self) -> &HashMap<Id, FunctionInfo> {
        &self.functions
    }

    pub fn apply_specialization(&mut self, specialization_info: &[(u32, SpecializationConstant)]) {
        self.constants = specialization::replace_specialization_instructions(
            specialization_info,
            self.constants.drain(..),
            &self.ids,
            self.bound,
        );

        for instruction in &self.constants {
            if let Some(id) = instruction.result_id() {
                if let Some(id_info) = self.ids.get_mut(&id) {
                    id_info.instruction = instruction.clone();
                    id_info.decorations.retain(|instruction| {
                        !matches!(
                            instruction,
                            Instruction::Decorate {
                                decoration: Decoration::SpecId { .. },
                                ..
                            }
                        )
                    });
                } else {
                    self.ids.insert(
                        id,
                        IdInfo {
                            instruction: instruction.clone(),
                            names: Vec::new(),
                            decorations: Vec::new(),
                            members: Vec::new(),
                        },
                    );
                    self.bound = self.bound.max(u32::from(id) + 1);
                }
            }
        }

        self.decorations.retain(|instruction| {
            !matches!(
                instruction,
                Instruction::Decorate {
                    decoration: Decoration::SpecId { .. },
                    ..
                }
            )
        });
    }
}

/// Used in SPIR-V to refer to the result of another instruction.
///
/// Ids are global across a module, and are always assigned by exactly one instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Id(u32);

impl Id {
    // Returns the raw numeric value of this Id.
    #[inline]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
}

impl From<Id> for u32 {
    #[inline]
    fn from(id: Id) -> u32 {
        id.as_raw()
    }
}

impl Display for Id {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(f, "%{}", self.0)
    }
}

/// Information associated with an `Id`.
#[derive(Clone, Debug)]
pub struct IdInfo {
    instruction: Instruction,
    names: Vec<Instruction>,
    decorations: Vec<Instruction>,
    members: Vec<StructMemberInfo>,
}

impl IdInfo {
    /// Returns the instruction that defines this `Id` with a `result_id` operand.
    #[inline]
    pub fn instruction(&self) -> &Instruction {
        &self.instruction
    }

    /// Returns all name debug instructions that target this `Id`.
    #[inline]
    pub fn names(&self) -> &[Instruction] {
        &self.names
    }

    /// Returns all decorate instructions that target this `Id`.
    #[inline]
    pub fn decorations(&self) -> &[Instruction] {
        &self.decorations
    }

    /// If this `Id` refers to a `TypeStruct`, returns information about each member of the struct.
    /// Empty otherwise.
    #[inline]
    pub fn members(&self) -> &[StructMemberInfo] {
        &self.members
    }
}

/// Information associated with a member of a `TypeStruct` instruction.
#[derive(Clone, Debug, Default)]
pub struct StructMemberInfo {
    names: Vec<Instruction>,
    decorations: Vec<Instruction>,
}

impl StructMemberInfo {
    /// Returns all name debug instructions that target this struct member.
    #[inline]
    pub fn names(&self) -> &[Instruction] {
        &self.names
    }

    /// Returns all decorate instructions that target this struct member.
    #[inline]
    pub fn decorations(&self) -> &[Instruction] {
        &self.decorations
    }
}

/// Information associated with a function.
#[derive(Clone, Debug)]
pub struct FunctionInfo {
    instructions: Vec<Instruction>,
    called_functions: HashSet<Id>,
    entry_point: Option<Instruction>,
    execution_modes: Vec<Instruction>,
}

impl FunctionInfo {
    /// Returns the instructions in the function.
    #[inline]
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Returns `Id`s of all functions that are called by this function.
    /// This may include recursive function calls.
    #[inline]
    pub fn called_functions(&self) -> &HashSet<Id> {
        &self.called_functions
    }

    /// Returns the `EntryPoint` instruction that targets this function, if there is one.
    #[inline]
    pub fn entry_point(&self) -> Option<&Instruction> {
        self.entry_point.as_ref()
    }

    /// Returns all execution mode instructions that target this function.
    #[inline]
    pub fn execution_modes(&self) -> &[Instruction] {
        &self.execution_modes
    }
}

fn iter_instructions(
    mut words: &[u32],
) -> impl Iterator<Item = Result<Instruction, ParseError>> + use<'_> {
    let mut index = 0;
    let next = move || -> Option<Result<Instruction, ParseError>> {
        if words.is_empty() {
            return None;
        }

        let word_count = (words[0] >> 16) as usize;
        assert!(word_count >= 1);

        if words.len() < word_count {
            return Some(Err(ParseError {
                instruction: index,
                word: words.len(),
                error: ParseErrors::UnexpectedEOF,
                words: words.to_owned(),
            }));
        }

        let mut reader = InstructionReader::new(&words[0..word_count], index);
        let instruction = match Instruction::parse(&mut reader) {
            Ok(x) => x,
            Err(err) => return Some(Err(err)),
        };

        if !reader.is_empty() {
            return Some(Err(reader.map_err(ParseErrors::LeftoverOperands)));
        }

        words = &words[word_count..];
        index += 1;
        Some(Ok(instruction))
    };

    std::iter::from_fn(next)
}

/// Helper type for parsing the words of an instruction.
#[derive(Debug)]
struct InstructionReader<'a> {
    words: &'a [u32],
    next_word: usize,
    instruction: usize,
}

impl<'a> InstructionReader<'a> {
    /// Constructs a new reader from a slice of words for a single instruction, including the
    /// opcode word. `instruction` is the number of the instruction currently being read, and
    /// is used for error reporting.
    fn new(words: &'a [u32], instruction: usize) -> Self {
        debug_assert!(!words.is_empty());
        Self {
            words,
            next_word: 0,
            instruction,
        }
    }

    /// Returns whether the reader has reached the end of the current instruction.
    fn is_empty(&self) -> bool {
        self.next_word >= self.words.len()
    }

    /// Converts the `ParseErrors` enum to the `ParseError` struct, adding contextual information.
    fn map_err(&self, error: ParseErrors) -> ParseError {
        ParseError {
            instruction: self.instruction,
            word: self.next_word - 1, // -1 because the word has already been read
            error,
            words: self.words.to_owned(),
        }
    }

    /// Returns the next word in the sequence.
    fn next_word(&mut self) -> Result<u32, ParseError> {
        let word = *self.words.get(self.next_word).ok_or_else(|| ParseError {
            instruction: self.instruction,
            word: self.next_word, // No -1 because we didn't advance yet
            error: ParseErrors::MissingOperands,
            words: self.words.to_owned(),
        })?;
        self.next_word += 1;

        Ok(word)
    }

    /*
    /// Returns the next two words as a single `u64`.
    #[inline]
    fn next_u64(&mut self) -> Result<u64, ParseError> {
        Ok(self.next_word()? as u64 | (self.next_word()? as u64) << 32)
    }
    */

    /// Reads a nul-terminated string.
    fn next_string(&mut self) -> Result<String, ParseError> {
        let mut bytes = Vec::new();
        loop {
            let word = self.next_word()?.to_le_bytes();

            if let Some(nul) = word.iter().position(|&b| b == 0) {
                bytes.extend(&word[0..nul]);
                break;
            } else {
                bytes.extend(word);
            }
        }
        String::from_utf8(bytes).map_err(|err| self.map_err(ParseErrors::FromUtf8Error(err)))
    }

    /// Reads all remaining words.
    fn remainder(&mut self) -> Vec<u32> {
        let vec = self.words[self.next_word..].to_owned();
        self.next_word = self.words.len();
        vec
    }
}

/// Error that can happen when reading a SPIR-V module.
#[derive(Clone, Debug)]
pub enum SpirvError {
    DuplicateId { id: Id },
    InvalidHeader,
    ParseError(ParseError),
}

impl Display for SpirvError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::DuplicateId { id } => write!(f, "id {} is assigned more than once", id,),
            Self::InvalidHeader => write!(f, "the SPIR-V module header is invalid"),
            Self::ParseError(_) => write!(f, "parse error"),
        }
    }
}

impl Error for SpirvError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ParseError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ParseError> for SpirvError {
    fn from(err: ParseError) -> Self {
        Self::ParseError(err)
    }
}

/// Error that can happen when parsing SPIR-V instructions into Rust data structures.
#[derive(Clone, Debug)]
pub struct ParseError {
    /// The instruction number the error happened at, starting from 0.
    pub instruction: usize,
    /// The word from the start of the instruction that the error happened at, starting from 0.
    pub word: usize,
    /// The error.
    pub error: ParseErrors,
    /// The words of the instruction.
    pub words: Vec<u32>,
}

impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "at instruction {}, word {}: {}",
            self.instruction, self.word, self.error,
        )
    }
}

impl Error for ParseError {}

/// Individual types of parse error that can happen.
#[derive(Clone, Debug)]
pub enum ParseErrors {
    FromUtf8Error(FromUtf8Error),
    LeftoverOperands,
    MissingOperands,
    UnexpectedEOF,
    UnknownEnumerant(&'static str, u32),
    UnknownOpcode(u16),
    UnknownSpecConstantOpcode(u16),
}

impl Display for ParseErrors {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::FromUtf8Error(_) => write!(f, "invalid UTF-8 in string literal"),
            Self::LeftoverOperands => write!(f, "unparsed operands remaining"),
            Self::MissingOperands => write!(
                f,
                "the instruction and its operands require more words than are present in the \
                instruction",
            ),
            Self::UnexpectedEOF => write!(f, "encountered unexpected end of file"),
            Self::UnknownEnumerant(ty, enumerant) => {
                write!(f, "invalid enumerant {} for enum {}", enumerant, ty)
            }
            Self::UnknownOpcode(opcode) => write!(f, "invalid instruction opcode {}", opcode),
            Self::UnknownSpecConstantOpcode(opcode) => {
                write!(f, "invalid spec constant instruction opcode {}", opcode)
            }
        }
    }
}

/// Converts SPIR-V bytes to words. If necessary, the byte order is swapped from little-endian
/// to native-endian.
pub fn bytes_to_words(bytes: &[u8]) -> Result<Cow<'_, [u32]>, SpirvBytesNotMultipleOf4> {
    // If the current target is little endian, and the slice already has the right size and
    // alignment, then we can just transmute the slice with bytemuck.
    #[cfg(target_endian = "little")]
    if let Ok(words) = bytemuck::try_cast_slice(bytes) {
        return Ok(Cow::Borrowed(words));
    }

    let (words, []) = bytes.as_chunks::<{ size_of::<u32>() }>() else {
        return Err(SpirvBytesNotMultipleOf4);
    };

    let words = words.iter().copied().map(u32::from_le_bytes).collect();

    Ok(Cow::Owned(words))
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SpirvBytesNotMultipleOf4;

impl Error for SpirvBytesNotMultipleOf4 {}

impl Display for SpirvBytesNotMultipleOf4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "the length of the provided slice is not a multiple of 4")
    }
}
