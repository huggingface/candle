use std::num::ParseIntError;

use safetensors::SafeTensorError;
use thiserror::Error;
use wasm_bindgen::{JsError, JsValue};

#[derive(Debug, Error)]
/// All errors the API can throw
pub enum GenericError {
    // /// The value cannot be used as a header during request header construction
    // #[error("Invalid header value {0}")]
    // InvalidHeaderValue(#[from] InvalidHeaderValue),
    /// Error parsing some range value
    #[error("Cannot parse int")]
    ParseIntError(#[from] ParseIntError),

    /// I/O Error
    #[error("I/O error {0}")]
    IoError(#[from] std::io::Error),

    /// We tried to download chunk too many times
    #[error("Too many retries: {0}")]
    TooManyRetries(Box<GenericError>),

    #[error("Javascript Error: {0:?}")]
    JsError(JsValue),

    #[error("Javascript Error Value: {0:?}")]
    JsValue(JsValue),

    #[error("Anyhow Error: {0}")]
    Anyhow(#[from] anyhow::Error),

    #[error("Candle Error: {0}")]
    CandleError(#[from] candle::Error),

    #[error("Safetensor Error: {0}")]
    SafetensorError(#[from] SafeTensorError),
}

impl From<JsError> for GenericError {
    fn from(value: JsError) -> Self {
        GenericError::JsError(value.into())
    }
}

impl From<String> for GenericError {
    fn from(value: String) -> Self {
        GenericError::JsError(value.into())
    }
}

impl From<JsValue> for GenericError {
    fn from(value: JsValue) -> Self {
        GenericError::JsValue(value)
    }
}

impl From<GenericError> for JsValue {
    fn from(value: GenericError) -> Self {
        match value {
            GenericError::JsError(val) => val,
            GenericError::JsValue(val) => val,
            e => JsValue::from_str(&e.to_string()),
        }
    }
}

impl From<&'static str> for GenericError {
    fn from(value: &'static str) -> Self {
        GenericError::Anyhow(anyhow::Error::msg(value))
    }
}

pub type GenericResult<T> = Result<T, GenericError>;
