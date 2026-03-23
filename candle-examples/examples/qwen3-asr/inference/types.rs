//! Public inference types.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub enum Batch<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> Batch<T> {
    pub fn one(v: T) -> Self {
        Self::One(v)
    }
}

impl<T: Clone> Batch<T> {
    pub fn broadcast(&self, n: usize, what: &'static str) -> Result<Vec<T>> {
        if n == 0 {
            return Ok(vec![]);
        }

        match self {
            Batch::One(v) => Ok(vec![v.clone(); n]),
            Batch::Many(vs) => {
                if vs.is_empty() {
                    bail!("{what} is empty");
                }

                if vs.len() == 1 && n > 1 {
                    let v0 = vs
                        .first()
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("{what} is empty"))?;
                    return Ok(vec![v0; n]);
                }

                if vs.len() != n {
                    bail!(
                        "batch size mismatch for {what}: expected={n} got={}",
                        vs.len()
                    );
                }

                Ok(vs.clone())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    pub context: Batch<String>,
    pub language: Batch<Option<String>>,
    pub return_timestamps: bool,
    pub max_new_tokens: usize,
    pub max_batch_size: usize,
    pub chunk_max_sec: Option<f32>,
    /// If true, bucket chunked audio by length before batching to reduce padding overhead.
    pub bucket_by_length: bool,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            context: Batch::one(String::new()),
            language: Batch::one(None),
            return_timestamps: false,
            max_new_tokens: 0,
            max_batch_size: 32,
            chunk_max_sec: None,
            bucket_by_length: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrTranscription {
    pub language: String,
    pub text: String,

    #[serde(default)]
    pub timestamps: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct StreamOptions {
    pub context: String,
    pub language: Option<String>,
    pub chunk_size_sec: f32,
    pub unfixed_chunk_num: usize,
    pub unfixed_token_num: usize,
    pub max_new_tokens: usize,
    /// Optional rolling audio window (seconds). When set, streaming decode only conditions on the
    /// most recent window of audio, rather than re-feeding all accumulated audio.
    pub audio_window_sec: Option<f32>,
    /// Optional rolling text context window (tokens). When set, streaming decode only conditions on
    /// the most recent decoded tokens, rather than the full transcript so far.
    pub text_window_tokens: Option<usize>,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            context: String::new(),
            language: None,
            chunk_size_sec: 2.0,
            unfixed_chunk_num: 2,
            unfixed_token_num: 5,
            max_new_tokens: 256,
            audio_window_sec: None,
            text_window_tokens: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Batch;

    #[test]
    fn test_batch_broadcast_one_repeats() -> anyhow::Result<()> {
        let b = Batch::one("x".to_string());
        let got = b.broadcast(3, "context")?;
        if got != vec!["x".to_string(), "x".to_string(), "x".to_string()] {
            anyhow::bail!("unexpected broadcast result: {got:?}");
        }
        Ok(())
    }

    #[test]
    fn test_batch_broadcast_many_len_one_repeats() -> anyhow::Result<()> {
        let b = Batch::Many(vec!["y".to_string()]);
        let got = b.broadcast(2, "context")?;
        if got != vec!["y".to_string(), "y".to_string()] {
            anyhow::bail!("unexpected broadcast result: {got:?}");
        }
        Ok(())
    }

    #[test]
    fn test_batch_broadcast_many_len_mismatch_errors() -> anyhow::Result<()> {
        let b = Batch::Many(vec!["a".to_string(), "b".to_string()]);
        let err = b
            .broadcast(3, "context")
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected error"))?;
        let msg = format!("{err:#}");
        if !msg.contains("batch size mismatch for context") {
            anyhow::bail!("unexpected error: {msg}");
        }
        Ok(())
    }
}
