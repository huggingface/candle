//! JSON-schema constraints for autoregressive decoding.
//!
//! The constraint deliberately operates on decoded token text: token ids are tokenizer-specific.

use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum SchemaCompileError {
    #[error("invalid JSON schema: {0}")]
    InvalidJson(String),
    #[error("unsupported JSON schema construct: {0}")]
    Unsupported(String),
    #[error("invalid JSON schema: {0}")]
    InvalidSchema(String),
}

#[derive(Clone, Debug)]
pub struct CompiledFsm {
    schema: Arc<Schema>,
}

#[derive(Clone, Debug)]
enum Schema {
    Any,
    Null,
    Bool,
    Number {
        integer: bool,
    },
    String,
    Array(Arc<Schema>),
    Object {
        properties: Vec<(String, Arc<Schema>)>,
        required: Vec<String>,
        additional: bool,
    },
    Enum(Vec<Value>),
}

/// Compile the supported, deliberately small JSON Schema subset into a shareable constraint.
///
/// Scalar string, boolean, and null `enum`/`const` values are supported. Numeric and composite
/// enum values are rejected rather than being approximated by a lexical prefix check.
pub fn compile_schema(schema: &str) -> Result<CompiledFsm, SchemaCompileError> {
    let value =
        serde_json::from_str(schema).map_err(|e| SchemaCompileError::InvalidJson(e.to_string()))?;
    Ok(CompiledFsm {
        schema: Arc::new(Schema::compile(&value)?),
    })
}

impl Schema {
    fn compile(value: &Value) -> Result<Self, SchemaCompileError> {
        let object = value.as_object().ok_or_else(|| {
            SchemaCompileError::InvalidSchema("the root must be an object".into())
        })?;
        for key in object.keys() {
            if !matches!(
                key.as_str(),
                "type"
                    | "properties"
                    | "required"
                    | "additionalProperties"
                    | "items"
                    | "enum"
                    | "const"
                    | "title"
                    | "description"
                    | "default"
            ) {
                return Err(SchemaCompileError::Unsupported(key.clone()));
            }
        }
        let has_enum = object.contains_key("enum");
        let has_const = object.contains_key("const");
        if (has_enum || has_const)
            && (has_enum && has_const
                || object.keys().any(|key| {
                    matches!(
                        key.as_str(),
                        "properties" | "required" | "additionalProperties" | "items"
                    )
                }))
        {
            return Err(SchemaCompileError::Unsupported(
                "combined enum/const constraints".into(),
            ));
        }
        if (has_enum || has_const) && object.contains_key("type") {
            let ty = object
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| SchemaCompileError::Unsupported("non-string type".into()))?;
            let values = if let Some(values) = object.get("enum") {
                values.as_array().ok_or_else(|| {
                    SchemaCompileError::InvalidSchema("enum must be an array".into())
                })?
            } else {
                std::slice::from_ref(object.get("const").expect("const is present"))
            };
            if !values.iter().all(|value| type_matches(ty, value)) {
                return Err(SchemaCompileError::InvalidSchema(
                    "type conflicts with enum/const".into(),
                ));
            }
        }
        if let Some(values) = object.get("enum") {
            let values = values
                .as_array()
                .ok_or_else(|| SchemaCompileError::InvalidSchema("enum must be an array".into()))?;
            if values.is_empty() {
                return Err(SchemaCompileError::InvalidSchema(
                    "enum must not be empty".into(),
                ));
            }
            if values
                .iter()
                .any(|value| value.is_array() || value.is_object())
            {
                return Err(SchemaCompileError::Unsupported("non-scalar enum".into()));
            }
            if values.iter().any(Value::is_number) {
                return Err(SchemaCompileError::Unsupported("numeric enum".into()));
            }
            return Ok(Self::Enum(values.clone()));
        }
        if let Some(value) = object.get("const") {
            if value.is_array() || value.is_object() {
                return Err(SchemaCompileError::Unsupported("non-scalar const".into()));
            }
            if value.is_number() {
                return Err(SchemaCompileError::Unsupported("numeric const".into()));
            }
            return Ok(Self::Enum(vec![value.clone()]));
        }
        let object_keywords = object.contains_key("properties")
            || object.contains_key("required")
            || object.contains_key("additionalProperties");
        let array_keywords = object.contains_key("items");
        let ty = match object.get("type") {
            None if !object_keywords && !array_keywords => return Ok(Self::Any),
            None => {
                return Err(SchemaCompileError::Unsupported(
                    "structural keywords without their matching type".into(),
                ))
            }
            Some(Value::String(ty)) => ty.as_str(),
            Some(_) => return Err(SchemaCompileError::Unsupported("non-string type".into())),
        };
        if (object_keywords && ty != "object") || (array_keywords && ty != "array") {
            return Err(SchemaCompileError::Unsupported(
                "structural keywords without their matching type".into(),
            ));
        }
        match ty {
            "null" => Ok(Self::Null),
            "boolean" => Ok(Self::Bool),
            "number" => Ok(Self::Number { integer: false }),
            "integer" => Ok(Self::Number { integer: true }),
            "string" => Ok(Self::String),
            "array" => Ok(Self::Array(Arc::new(match object.get("items") {
                Some(items) => Self::compile(items)?,
                None => Self::Any,
            }))),
            "object" => {
                let properties: Vec<(String, Arc<Schema>)> = match object.get("properties") {
                    None => vec![],
                    Some(properties) => properties
                        .as_object()
                        .ok_or_else(|| {
                            SchemaCompileError::InvalidSchema("properties must be an object".into())
                        })?
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), Arc::new(Self::compile(v)?))))
                        .collect::<Result<_, SchemaCompileError>>()?,
                };
                let required: Vec<String> = object
                    .get("required")
                    .map(|v| {
                        v.as_array()
                            .ok_or_else(|| {
                                SchemaCompileError::InvalidSchema(
                                    "required must be an array".into(),
                                )
                            })
                            .and_then(|v| {
                                v.iter()
                                    .map(|v| {
                                        v.as_str().map(str::to_owned).ok_or_else(|| {
                                            SchemaCompileError::InvalidSchema(
                                                "required entries must be strings".into(),
                                            )
                                        })
                                    })
                                    .collect()
                            })
                    })
                    .transpose()?
                    .unwrap_or_default();
                let additional = match object.get("additionalProperties") {
                    None => true,
                    Some(Value::Bool(additional)) => *additional,
                    Some(_) => {
                        return Err(SchemaCompileError::Unsupported(
                            "non-boolean additionalProperties".into(),
                        ))
                    }
                };
                if !additional
                    && required
                        .iter()
                        .any(|key| !properties.iter().any(|(name, _)| name == key))
                {
                    return Err(SchemaCompileError::InvalidSchema(
                        "required property is not declared".into(),
                    ));
                }
                Ok(Self::Object {
                    properties,
                    required,
                    additional,
                })
            }
            other => Err(SchemaCompileError::Unsupported(format!("type {other}"))),
        }
    }

    fn matches(&self, value: &Value) -> bool {
        match self {
            Self::Any => true,
            Self::Null => value.is_null(),
            Self::Bool => value.is_boolean(),
            Self::String => value.is_string(),
            Self::Number { integer } => {
                value.as_f64().is_some_and(|n| !integer || n.fract() == 0.0)
            }
            Self::Array(item) => value
                .as_array()
                .is_some_and(|a| a.iter().all(|v| item.matches(v))),
            Self::Enum(values) => values.contains(value),
            Self::Object {
                properties,
                required,
                additional,
            } => value.as_object().is_some_and(|o| {
                required.iter().all(|k| o.contains_key(k))
                    && o.iter().all(|(k, v)| {
                        properties
                            .iter()
                            .find(|(name, _)| name == k)
                            .map_or(*additional, |(_, s)| s.matches(v))
                    })
            }),
        }
    }
}

/// Per-request state for a compiled schema.
pub struct FsmLogitProcessor {
    cursor: Cursor,
    text: String,
}
impl FsmLogitProcessor {
    pub fn new(fsm: Arc<CompiledFsm>) -> Self {
        Self {
            cursor: Cursor::new(fsm.schema.clone()),
            text: String::new(),
        }
    }
    pub fn allowed_token_ids(
        &self,
        vocab_size: usize,
        eos_tokens: &[u32],
        decode_one: impl Fn(u32) -> String,
    ) -> Vec<u32> {
        (0..vocab_size as u32)
            .filter(|id| {
                if eos_tokens.contains(id) {
                    self.is_complete()
                } else {
                    self.accepts(&decode_one(*id))
                }
            })
            .collect()
    }
    /// Advance after sampling the actual token (not necessarily the best-scoring one).
    pub fn commit(&mut self, token_text: &str) {
        let mut cursor = self.cursor.clone();
        let accepted = cursor.apply(token_text);
        debug_assert!(accepted, "committed text must be accepted by the cursor");
        if accepted {
            self.cursor = cursor;
            self.text.push_str(token_text);
        }
    }
    pub fn is_complete(&self) -> bool {
        self.cursor.is_complete()
    }
    pub fn text(&self) -> &str {
        &self.text
    }
    fn accepts(&self, token: &str) -> bool {
        if token.is_empty() {
            return false;
        }
        let mut cursor = self.cursor.clone();
        cursor.apply(token)
    }
}

#[derive(Clone)]
struct Cursor {
    root: Arc<Schema>,
    state: Root,
    frames: Vec<Frame>,
    scalar: Option<Scalar>,
}
#[derive(Clone, Copy, PartialEq)]
enum Root {
    Value,
    Done,
}
#[derive(Clone)]
enum Frame {
    Object(Object),
    Array(Array),
}
#[derive(Clone)]
struct Object {
    schema: Arc<Schema>,
    seen: Vec<String>,
    key: String,
    state: Obj,
}
#[derive(Clone, Copy, PartialEq)]
enum Obj {
    KeyOrEnd,
    KeyRequired,
    Key,
    Colon,
    Value,
    CommaOrEnd,
}
#[derive(Clone)]
struct Array {
    schema: Arc<Schema>,
    state: Arr,
}
#[derive(Clone, Copy, PartialEq)]
enum Arr {
    ValueOrEnd,
    ValueRequired,
    CommaOrEnd,
}
#[derive(Clone)]
enum Scalar {
    Key(Text),
    String {
        schema: Arc<Schema>,
        text: Text,
    },
    Number {
        schema: Arc<Schema>,
        state: Number,
        raw: String,
    },
    Literal {
        expected: &'static str,
        at: usize,
    },
}
#[derive(Clone, Default)]
struct Text {
    value: String,
    escape: Escape,
}
#[derive(Clone, Default)]
enum Escape {
    #[default]
    None,
    Slash,
    Unicode(String),
    HighSurrogateSlash(u16),
    HighSurrogateU(u16),
    LowSurrogate(u16, String),
}
#[derive(Clone, Copy)]
enum Number {
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDigits,
}

impl Cursor {
    fn new(root: Arc<Schema>) -> Self {
        Self {
            root,
            state: Root::Value,
            frames: vec![],
            scalar: None,
        }
    }
    fn apply(&mut self, text: &str) -> bool {
        self.state != Root::Done && text.chars().all(|c| self.step(c))
    }
    fn is_complete(&self) -> bool {
        self.frames.is_empty()
            && match (&self.state, &self.scalar) {
                (Root::Done, None) => true,
                (Root::Value, Some(Scalar::Number { schema, state, raw })) => {
                    state.complete() && number_matches(schema, *state, raw)
                }
                _ => false,
            }
    }
    fn step(&mut self, c: char) -> bool {
        if let Some(scalar) = self.scalar.take() {
            return self.scalar_step(scalar, c);
        }
        self.structural(c)
    }
    fn scalar_step(&mut self, mut scalar: Scalar, c: char) -> bool {
        match &mut scalar {
            Scalar::Key(text) => match text.push(c) {
                TextResult::More => {
                    let allowed = matches!(self.frames.last(), Some(Frame::Object(object)) if object.key_prefix_allowed(&text.value));
                    if allowed {
                        self.scalar = Some(scalar);
                    }
                    allowed
                }
                TextResult::Done => {
                    let Some(Frame::Object(object)) = self.frames.last_mut() else {
                        return false;
                    };
                    if !object.key_allowed(&text.value) {
                        return false;
                    }
                    object.key = text.value.clone();
                    object.state = Obj::Colon;
                    true
                }
                TextResult::Invalid => false,
            },
            Scalar::String { schema, text } => match text.push(c) {
                TextResult::More => {
                    if string_prefix_possible(schema, &text.value) {
                        self.scalar = Some(scalar);
                        true
                    } else {
                        false
                    }
                }
                TextResult::Done => {
                    if !schema.matches(&Value::String(text.value.clone())) {
                        return false;
                    }
                    self.value_done()
                }
                TextResult::Invalid => false,
            },
            Scalar::Literal { expected, at } => {
                if *at >= expected.len() || c != expected.as_bytes()[*at] as char {
                    return false;
                }
                *at += 1;
                if *at == expected.len() {
                    self.value_done()
                } else {
                    self.scalar = Some(scalar);
                    true
                }
            }
            Scalar::Number { schema, state, raw } => match state.advance(c) {
                Some(next) => {
                    *state = next;
                    raw.push(c);
                    if number_prefix_possible(schema, *state, raw) {
                        self.scalar = Some(scalar);
                        true
                    } else {
                        false
                    }
                }
                None if state.complete() && delimiter(c) => {
                    let valid = number_matches(schema, *state, raw);
                    if !valid || !self.value_done() {
                        return false;
                    }
                    self.step(c)
                }
                None => false,
            },
        }
    }
    fn structural(&mut self, c: char) -> bool {
        if json_whitespace(c) {
            return true;
        }
        if self.frames.is_empty() {
            return match self.state {
                Root::Value => self.start(self.root.clone(), c),
                Root::Done => false,
            };
        }
        match self.frames.last().cloned().expect("non-empty") {
            Frame::Object(object) => match object.state {
                Obj::KeyOrEnd => {
                    if c == '}' {
                        self.close_object()
                    } else if c == '"' && object.key_prefix_allowed("") {
                        self.set_object(Obj::Key, String::new());
                        self.scalar = Some(Scalar::Key(Text::default()));
                        true
                    } else {
                        false
                    }
                }
                Obj::KeyRequired => {
                    if c == '"' && object.key_prefix_allowed("") {
                        self.set_object(Obj::Key, String::new());
                        self.scalar = Some(Scalar::Key(Text::default()));
                        true
                    } else {
                        false
                    }
                }
                Obj::Colon => {
                    if c == ':' {
                        self.set_object(Obj::Value, object.key);
                        true
                    } else {
                        false
                    }
                }
                Obj::Value => self.start(object.value_schema(), c),
                Obj::CommaOrEnd => {
                    if c == ',' {
                        self.set_object(Obj::KeyRequired, String::new());
                        true
                    } else if c == '}' {
                        self.close_object()
                    } else {
                        false
                    }
                }
                Obj::Key => false,
            },
            Frame::Array(array) => match array.state {
                Arr::ValueOrEnd => {
                    if c == ']' {
                        self.close_array()
                    } else {
                        self.start(array.item_schema(), c)
                    }
                }
                Arr::ValueRequired => self.start(array.item_schema(), c),
                Arr::CommaOrEnd => {
                    if c == ',' {
                        self.set_array(Arr::ValueRequired);
                        true
                    } else if c == ']' {
                        self.close_array()
                    } else {
                        false
                    }
                }
            },
        }
    }
    fn start(&mut self, schema: Arc<Schema>, c: char) -> bool {
        if json_whitespace(c) {
            return true;
        }
        match c {
            '{' if schema_object(&schema) => {
                self.frames.push(Frame::Object(Object {
                    schema,
                    seen: vec![],
                    key: String::new(),
                    state: Obj::KeyOrEnd,
                }));
                true
            }
            '[' if schema_array(&schema) => {
                self.frames.push(Frame::Array(Array {
                    schema,
                    state: Arr::ValueOrEnd,
                }));
                true
            }
            '"' if schema_string(&schema) => {
                self.scalar = Some(Scalar::String {
                    schema,
                    text: Text::default(),
                });
                true
            }
            't' => self.literal(schema, "true", 1),
            'f' => self.literal(schema, "false", 1),
            'n' => self.literal(schema, "null", 1),
            '-' => self.number(schema, Number::Minus, "-"),
            '0' => self.number(schema, Number::Zero, "0"),
            '1'..='9' => self.number(schema, Number::Int, &c.to_string()),
            _ => false,
        }
    }
    fn literal(&mut self, schema: Arc<Schema>, expected: &'static str, at: usize) -> bool {
        if schema_allows_literal(&schema, expected) {
            self.scalar = Some(Scalar::Literal { expected, at });
            true
        } else {
            false
        }
    }
    fn number(&mut self, schema: Arc<Schema>, state: Number, raw: &str) -> bool {
        if schema_allows_number(&schema) && number_prefix_possible(&schema, state, raw) {
            self.scalar = Some(Scalar::Number {
                schema,
                state,
                raw: raw.into(),
            });
            true
        } else {
            false
        }
    }
    fn value_done(&mut self) -> bool {
        if let Some(frame) = self.frames.last_mut() {
            match frame {
                Frame::Object(object) if object.state == Obj::Value => {
                    object.seen.push(std::mem::take(&mut object.key));
                    object.state = Obj::CommaOrEnd;
                    true
                }
                Frame::Array(array)
                    if matches!(array.state, Arr::ValueOrEnd | Arr::ValueRequired) =>
                {
                    array.state = Arr::CommaOrEnd;
                    true
                }
                _ => false,
            }
        } else if self.state == Root::Value {
            self.state = Root::Done;
            true
        } else {
            false
        }
    }
    fn close_object(&mut self) -> bool {
        let Some(Frame::Object(object)) = self.frames.pop() else {
            return false;
        };
        if !object.valid() {
            return false;
        }
        self.value_done()
    }
    fn close_array(&mut self) -> bool {
        let Some(Frame::Array(_)) = self.frames.pop() else {
            return false;
        };
        self.value_done()
    }
    fn set_object(&mut self, state: Obj, key: String) {
        if let Some(Frame::Object(object)) = self.frames.last_mut() {
            object.state = state;
            object.key = key;
        }
    }
    fn set_array(&mut self, state: Arr) {
        if let Some(Frame::Array(array)) = self.frames.last_mut() {
            array.state = state;
        }
    }
}

enum TextResult {
    More,
    Done,
    Invalid,
}
impl Text {
    fn push(&mut self, c: char) -> TextResult {
        match &mut self.escape {
            Escape::None => match c {
                '"' => TextResult::Done,
                '\\' => {
                    self.escape = Escape::Slash;
                    TextResult::More
                }
                c if c >= ' ' => {
                    self.value.push(c);
                    TextResult::More
                }
                _ => TextResult::Invalid,
            },
            Escape::Slash => {
                match c {
                    '"' | '\\' | '/' => self.value.push(c),
                    'b' => self.value.push('\u{8}'),
                    'f' => self.value.push('\u{c}'),
                    'n' => self.value.push('\n'),
                    'r' => self.value.push('\r'),
                    't' => self.value.push('\t'),
                    'u' => {
                        self.escape = Escape::Unicode(String::new());
                        return TextResult::More;
                    }
                    _ => return TextResult::Invalid,
                };
                self.escape = Escape::None;
                TextResult::More
            }
            Escape::Unicode(hex) => {
                if !c.is_ascii_hexdigit() {
                    return TextResult::Invalid;
                }
                hex.push(c);
                if hex.len() == 4 {
                    let code = u32::from_str_radix(hex, 16).unwrap();
                    match code {
                        0xd800..=0xdbff => {
                            self.escape = Escape::HighSurrogateSlash(code as u16);
                        }
                        0xdc00..=0xdfff => return TextResult::Invalid,
                        _ => {
                            self.value.push(char::from_u32(code).expect("valid scalar"));
                            self.escape = Escape::None;
                        }
                    }
                }
                TextResult::More
            }
            Escape::HighSurrogateSlash(high) => {
                if c == '\\' {
                    self.escape = Escape::HighSurrogateU(*high);
                    TextResult::More
                } else {
                    TextResult::Invalid
                }
            }
            Escape::HighSurrogateU(high) => {
                if c == 'u' {
                    self.escape = Escape::LowSurrogate(*high, String::new());
                    TextResult::More
                } else {
                    TextResult::Invalid
                }
            }
            Escape::LowSurrogate(high, hex) => {
                if !c.is_ascii_hexdigit() {
                    return TextResult::Invalid;
                }
                hex.push(c);
                if hex.len() == 4 {
                    let low = u16::from_str_radix(hex, 16).unwrap();
                    if !(0xdc00..=0xdfff).contains(&low) {
                        return TextResult::Invalid;
                    }
                    let scalar =
                        0x1_0000 + (((*high as u32 - 0xd800) << 10) | (low as u32 - 0xdc00));
                    self.value
                        .push(char::from_u32(scalar).expect("valid surrogate pair"));
                    self.escape = Escape::None;
                }
                TextResult::More
            }
        }
    }
}
impl Number {
    fn advance(self, c: char) -> Option<Self> {
        match (self, c) {
            (Self::Minus, '0') => Some(Self::Zero),
            (Self::Minus, '1'..='9') => Some(Self::Int),
            (Self::Zero, '.') | (Self::Int, '.') => Some(Self::Dot),
            (Self::Zero, 'e' | 'E') | (Self::Int, 'e' | 'E') | (Self::Frac, 'e' | 'E') => {
                Some(Self::Exp)
            }
            (Self::Int, '0'..='9') => Some(Self::Int),
            (Self::Dot, '0'..='9') => Some(Self::Frac),
            (Self::Frac, '0'..='9') => Some(Self::Frac),
            (Self::Exp, '+' | '-') => Some(Self::ExpSign),
            (Self::Exp, '0'..='9') | (Self::ExpSign, '0'..='9') => Some(Self::ExpDigits),
            (Self::ExpDigits, '0'..='9') => Some(Self::ExpDigits),
            _ => None,
        }
    }
    fn complete(self) -> bool {
        matches!(self, Self::Zero | Self::Int | Self::Frac | Self::ExpDigits)
    }
}
impl Object {
    fn key_prefix_allowed(&self, key: &str) -> bool {
        match &*self.schema {
            Schema::Object {
                properties,
                additional,
                ..
            } => *additional || properties.iter().any(|(name, _)| name.starts_with(key)),
            Schema::Any => true,
            _ => false,
        }
    }
    fn key_allowed(&self, key: &str) -> bool {
        match &*self.schema {
            Schema::Object {
                properties,
                additional,
                ..
            } => *additional || properties.iter().any(|(name, _)| name == key),
            Schema::Any => true,
            _ => false,
        }
    }
    fn value_schema(&self) -> Arc<Schema> {
        match &*self.schema {
            Schema::Object {
                properties,
                additional,
                ..
            } => properties
                .iter()
                .find(|(name, _)| name == &self.key)
                .map(|(_, s)| s.clone())
                .unwrap_or_else(|| {
                    if *additional {
                        Arc::new(Schema::Any)
                    } else {
                        unreachable!()
                    }
                }),
            Schema::Any => Arc::new(Schema::Any),
            _ => unreachable!(),
        }
    }
    fn valid(&self) -> bool {
        match &*self.schema {
            Schema::Object { required, .. } => required.iter().all(|key| self.seen.contains(key)),
            Schema::Any => true,
            _ => false,
        }
    }
}
impl Array {
    fn item_schema(&self) -> Arc<Schema> {
        match &*self.schema {
            Schema::Array(item) => Arc::clone(item),
            Schema::Any => Arc::new(Schema::Any),
            _ => unreachable!(),
        }
    }
}
fn delimiter(c: char) -> bool {
    json_whitespace(c) || matches!(c, ',' | ']' | '}')
}
fn json_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}
fn schema_object(s: &Schema) -> bool {
    matches!(s, Schema::Object { .. } | Schema::Any)
}
fn type_matches(ty: &str, value: &Value) -> bool {
    match ty {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "number" => value.is_number(),
        "integer" => value.as_f64().is_some_and(|number| number.fract() == 0.0),
        "string" => value.is_string(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}
fn schema_array(s: &Schema) -> bool {
    matches!(s, Schema::Array(_) | Schema::Any)
}
fn schema_string(s: &Schema) -> bool {
    matches!(s, Schema::String | Schema::Any)
        || matches!(s, Schema::Enum(values) if values.iter().any(Value::is_string))
}
fn schema_allows_literal(s: &Schema, literal: &str) -> bool {
    matches!(s, Schema::Any)
        || matches!(s, Schema::Bool if literal == "true" || literal == "false")
        || matches!(s, Schema::Null if literal == "null")
        || matches!(s, Schema::Enum(values) if values.iter().any(|value| match literal {
            "true" => value == &Value::Bool(true),
            "false" => value == &Value::Bool(false),
            "null" => value.is_null(),
            _ => false,
        }))
}
fn schema_allows_number(s: &Schema) -> bool {
    matches!(s, Schema::Number { .. } | Schema::Any)
}
fn string_prefix_possible(schema: &Schema, prefix: &str) -> bool {
    match schema {
        Schema::Enum(values) => values
            .iter()
            .filter_map(Value::as_str)
            .any(|value| value.starts_with(prefix)),
        _ => true,
    }
}
fn number_prefix_possible(schema: &Schema, state: Number, prefix: &str) -> bool {
    if !state.complete() || !json_number_is_finite(prefix) {
        return !state.complete();
    }
    !matches!(schema, Schema::Number { integer: true })
        || !matches!(state, Number::ExpDigits)
        || number_is_integer(prefix)
        || !has_negative_exponent(prefix)
}
fn has_negative_exponent(raw: &str) -> bool {
    raw.find(['e', 'E'])
        .is_some_and(|index| raw[index + 1..].starts_with('-'))
}
fn number_matches(s: &Schema, _state: Number, raw: &str) -> bool {
    if !json_number_is_finite(raw) {
        return false;
    }
    match s {
        Schema::Any => true,
        Schema::Number { integer } => !integer || number_is_integer(raw),
        _ => false,
    }
}
fn json_number_is_finite(raw: &str) -> bool {
    serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|value| value.as_f64())
        .is_some_and(f64::is_finite)
}
fn number_is_integer(raw: &str) -> bool {
    let (mantissa, exponent) = match raw.find(['e', 'E']) {
        Some(index) => (&raw[..index], &raw[index + 1..]),
        None => (raw, "0"),
    };
    let mantissa = mantissa.strip_prefix('-').unwrap_or(mantissa);
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    if digits.bytes().all(|byte| byte == b'0') {
        return true;
    }
    let Ok(exponent) = exponent.parse::<i128>() else {
        return false;
    };
    let trailing_zeroes = digits
        .bytes()
        .rev()
        .take_while(|byte| *byte == b'0')
        .count() as i128;
    let decimal_places = fraction.len() as i128 - exponent;
    decimal_places <= 0 || trailing_zeroes >= decimal_places
}
