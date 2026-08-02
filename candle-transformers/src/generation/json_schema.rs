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
pub struct CompiledFsm { schema: Schema }

#[derive(Clone, Debug)]
enum Schema {
    Any, Null, Bool, Number { integer: bool }, String,
    Array(Box<Schema>),
    Object { properties: Vec<(String, Schema)>, required: Vec<String>, additional: bool },
    Enum(Vec<Value>),
}

/// Compile the supported, deliberately small JSON Schema subset into a shareable constraint.
pub fn compile_schema(schema: &str) -> Result<CompiledFsm, SchemaCompileError> {
    let value = serde_json::from_str(schema).map_err(|e| SchemaCompileError::InvalidJson(e.to_string()))?;
    Ok(CompiledFsm { schema: Schema::compile(&value)? })
}

impl Schema {
    fn compile(value: &Value) -> Result<Self, SchemaCompileError> {
        let object = value.as_object().ok_or_else(|| SchemaCompileError::InvalidSchema("the root must be an object".into()))?;
        for key in object.keys() {
            if !matches!(key.as_str(), "type" | "properties" | "required" | "additionalProperties" | "items" | "enum" | "const" | "title" | "description" | "default") {
                return Err(SchemaCompileError::Unsupported(key.clone()));
            }
        }
        if let Some(values) = object.get("enum") { return Ok(Self::Enum(values.as_array().ok_or_else(|| SchemaCompileError::InvalidSchema("enum must be an array".into()))?.clone())); }
        if let Some(value) = object.get("const") { return Ok(Self::Enum(vec![value.clone()])); }
        let ty = object.get("type").and_then(Value::as_str).unwrap_or("any");
        match ty {
            "any" => Ok(Self::Any), "null" => Ok(Self::Null), "boolean" => Ok(Self::Bool),
            "number" => Ok(Self::Number { integer: false }), "integer" => Ok(Self::Number { integer: true }), "string" => Ok(Self::String),
            "array" => Ok(Self::Array(Box::new(Self::compile(object.get("items").ok_or_else(|| SchemaCompileError::InvalidSchema("array requires items".into()))?)?))),
            "object" => {
                let properties = object.get("properties").and_then(Value::as_object).ok_or_else(|| SchemaCompileError::InvalidSchema("object requires properties".into()))?
                    .iter().map(|(k, v)| Ok((k.clone(), Self::compile(v)?))).collect::<Result<_, SchemaCompileError>>()?;
                let required = object.get("required").map(|v| v.as_array().ok_or_else(|| SchemaCompileError::InvalidSchema("required must be an array".into()))
                    .and_then(|v| v.iter().map(|v| v.as_str().map(str::to_owned).ok_or_else(|| SchemaCompileError::InvalidSchema("required entries must be strings".into()))).collect())).transpose()?.unwrap_or_default();
                let additional = object.get("additionalProperties").and_then(Value::as_bool).unwrap_or(true);
                Ok(Self::Object { properties, required, additional })
            }
            other => Err(SchemaCompileError::Unsupported(format!("type {other}"))),
        }
    }

    fn matches(&self, value: &Value) -> bool { match self {
        Self::Any => true, Self::Null => value.is_null(), Self::Bool => value.is_boolean(), Self::String => value.is_string(),
        Self::Number { integer } => value.as_f64().is_some_and(|n| !integer || n.fract() == 0.0),
        Self::Array(item) => value.as_array().is_some_and(|a| a.iter().all(|v| item.matches(v))),
        Self::Enum(values) => values.contains(value),
        Self::Object { properties, required, additional } => value.as_object().is_some_and(|o| {
            required.iter().all(|k| o.contains_key(k)) && o.iter().all(|(k, v)| properties.iter().find(|(name, _)| name == k).map_or(*additional, |(_, s)| s.matches(v)))
        }),
    }}
}

/// Per-request state for a compiled schema.
pub struct FsmLogitProcessor { fsm: Arc<CompiledFsm>, text: String }
impl FsmLogitProcessor {
    pub fn new(fsm: Arc<CompiledFsm>) -> Self { Self { fsm, text: String::new() } }
    pub fn allowed_token_ids(&self, vocab_size: usize, eos_tokens: &[u32], decode_one: impl Fn(u32) -> String) -> Vec<u32> {
        (0..vocab_size as u32).filter(|id| {
            if eos_tokens.contains(id) { self.is_complete() } else { self.accepts(&decode_one(*id)) }
        }).collect()
    }
    /// Advance after sampling the actual token (not necessarily the best-scoring one).
    pub fn commit(&mut self, token_text: &str) { self.text.push_str(token_text); }
    pub fn is_complete(&self) -> bool { serde_json::from_str(&self.text).is_ok_and(|v| self.fsm.schema.matches(&v)) }
    pub fn text(&self) -> &str { &self.text }
    fn accepts(&self, token: &str) -> bool {
        // A complete root JSON value cannot be followed by another token (except EOS).
        if self.is_complete() { return false; }
        let candidate = format!("{}{}", self.text, token);
        if serde_json::from_str::<Value>(&candidate).is_ok_and(|v| self.fsm.schema.matches(&v)) { return true; }
        json_prefix(&candidate)
    }
}

// Reject impossible JSON structure while leaving unfinished lexical values available to later tokens.
fn json_prefix(text: &str) -> bool {
    if let Some(first) = text.chars().find(|c| !c.is_whitespace()) {
        if !matches!(first, '{' | '[' | '"' | 't' | 'f' | 'n' | '-' | '0'..='9') { return false; }
    }
    let mut stack = Vec::new(); let mut string = false; let mut escape = false;
    for c in text.chars() {
        if string { if escape { escape = false } else if c == '\\' { escape = true } else if c == '"' { string = false }; continue; }
        match c { '"' => string = true, '{' => stack.push('}'), '[' => stack.push(']'), '}' | ']' => if stack.pop() != Some(c) { return false }, _ => {} }
    }
    true
}
