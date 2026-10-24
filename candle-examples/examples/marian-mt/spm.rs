//! Marian tokenizer from sentencepiece and vocab in `Helsinki-NLP/opus-mt-*` repos.
//!
//! Same functionality as the `transformers` `SpmConverter`, just without the
//! python dependency.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
use tokenizers::models::unigram::Unigram;
use tokenizers::normalizers::replace::ReplacePattern;
use tokenizers::normalizers::{Precompiled, Replace, Sequence};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::{AddedToken, Tokenizer};

/// Placeholder for ids that are in `vocab.json` but has no matching sentence piece.
const UNUSED_PIECE: &str = "<NIL>";
const UNUSED_SCORE: f64 = -100.;

/// Special Marian tokens. See `MarianTokenizer` in `transformers`.
const SPECIAL_TOKENS: [&str; 3] = ["</s>", "<unk>", "<pad>"];

/// Minimal protobuf reader
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

enum Value<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            let Some(byte) = self.buf.get(self.pos).copied() else {
                bail!("protobuf: unexpected end of data")
            };
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 64 {
                bail!("protobuf: varint is too large")
            }
        }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = match self.pos.checked_add(len) {
            Some(end) if end <= self.buf.len() => end,
            _ => bail!("protobuf: unexpected end of data"),
        };
        let bytes = &self.buf[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn next_field(&mut self) -> Result<(u64, Value<'a>)> {
        let key = self.varint()?;
        let value = match key & 7 {
            0 => Value::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Value::Fixed64
            }
            2 => {
                let len = self.varint()? as usize;
                Value::Bytes(self.take(len)?)
            }
            5 => Value::Fixed32(u32::from_le_bytes(self.take(4)?.try_into()?)),
            wire_type => bail!("protobuf: unsupported wire type - {wire_type}"),
        };
        Ok((key >> 3, value))
    }
}

#[derive(Default)]
struct SpmModel {
    /// `ModelProto.pieces`, in SentencePiece order rather than in `vocab.json` order.
    pieces: Vec<(String, f64)>,
    /// `TrainerSpec.unk_id`.
    unk_id: i64,
    /// `TrainerSpec.byte_fallback`.
    byte_fallback: bool,
    /// `NormalizerSpec.precompiled_charsmap`.
    precompiled_charsmap: Vec<u8>,
}

impl SpmModel {
    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let data = std::fs::read(path)
            .with_context(|| format!("cannot read the spm model {}", path.display()))?;
        Self::from_slice(&data)
            .with_context(|| format!("cannot parse the spm model {}", path.display()))
    }

    fn from_slice(data: &[u8]) -> Result<Self> {
        let mut model = Self::default();
        let mut reader = Reader::new(data);
        while !reader.is_empty() {
            match reader.next_field()? {
                // ModelProto.pieces
                (1, Value::Bytes(bytes)) => model.pieces.push(Self::piece(bytes)?),
                // ModelProto.trainer_spec
                (2, Value::Bytes(bytes)) => {
                    let mut reader = Reader::new(bytes);
                    while !reader.is_empty() {
                        match reader.next_field()? {
                            // TrainerSpec.byte_fallback
                            (35, Value::Varint(v)) => model.byte_fallback = v != 0,
                            // TrainerSpec.unk_id
                            (40, Value::Varint(v)) => model.unk_id = v as i64,
                            _ => {}
                        }
                    }
                }
                // ModelProto.normalizer_spec
                (3, Value::Bytes(bytes)) => {
                    let mut reader = Reader::new(bytes);
                    while !reader.is_empty() {
                        // NormalizerSpec.precompiled_charsmap
                        if let (2, Value::Bytes(bytes)) = reader.next_field()? {
                            model.precompiled_charsmap = bytes.to_vec()
                        }
                    }
                }
                _ => {}
            }
        }
        if model.pieces.is_empty() {
            bail!("no sentencepiece pieces found")
        }
        Ok(model)
    }

    /// `ModelProto.SentencePiece`. We only need the piece and score here.
    fn piece(data: &[u8]) -> Result<(String, f64)> {
        let (mut piece, mut score) = (None, 0f64);
        let mut reader = Reader::new(data);
        while !reader.is_empty() {
            match reader.next_field()? {
                (1, Value::Bytes(bytes)) => piece = Some(String::from_utf8(bytes.to_vec())?),
                (2, Value::Fixed32(v)) => score = f32::from_bits(v) as f64,
                _ => {}
            }
        }
        match piece {
            Some(piece) => Ok((piece, score)),
            None => bail!("sentencepiece piece without a `piece` field"),
        }
    }
}

/// Builds a tokenizer out of a SentencePiece model and the `vocab.json` that ships alongside it.
///
/// Marian uses a single `vocab.json` for both translation directions, only the spm model differs
/// between the encoder and the decoder side.
pub fn tokenizer<P: AsRef<Path>, Q: AsRef<Path>>(spm_file: P, vocab_file: Q) -> Result<Tokenizer> {
    let spm = SpmModel::from_file(spm_file)?;
    let vocab_file = vocab_file.as_ref();
    let vocab: HashMap<String, usize> = {
        let vocab = std::fs::read(vocab_file)
            .with_context(|| format!("cannot read the vocab {}", vocab_file.display()))?;
        serde_json::from_slice(&vocab)
            .with_context(|| format!("cannot parse the vocab {}", vocab_file.display()))?
    };
    let Some(vocab_size) = vocab.values().max().map(|max| max + 1) else {
        bail!("{} is empty", vocab_file.display())
    };

    // The spm pieces are a subset of the vocab and are not in vocab order, so the vocabulary is
    // indexed by the ids from vocab.json. Ids that no piece maps to are left unused.
    let mut pieces = vec![(UNUSED_PIECE.to_string(), UNUSED_SCORE); vocab_size];
    for (piece, score) in spm.pieces.into_iter() {
        // Pieces that are missing from vocab.json are dropped. For example `<s>`.
        if let Some(&index) = vocab.get(&piece) {
            pieces[index] = (piece, score)
        }
    }

    // Special tokens have to be part of the unigram vocabulary for the ids to resolve.
    // For example `<pad>`.
    let special_tokens: Vec<(&str, usize)> = SPECIAL_TOKENS
        .iter()
        .filter_map(|token| vocab.get(*token).map(|&index| (*token, index)))
        .collect();
    for (token, index) in special_tokens.iter() {
        if pieces[*index].0 == UNUSED_PIECE {
            pieces[*index] = (token.to_string(), 0.)
        }
    }

    // Prefer the id that vocab.json gives to the unk token.
    let unk_id = match vocab.get("<unk>") {
        Some(&unk_id) => unk_id,
        None if spm.unk_id >= 0 && (spm.unk_id as usize) < vocab_size => spm.unk_id as usize,
        None => bail!("no <unk> token in {}", vocab_file.display()),
    };

    let model = Unigram::from(pieces, Some(unk_id), spm.byte_fallback)
        .map_err(anyhow::Error::msg)
        .context("cannot build the unigram model")?;
    let mut tokenizer = Tokenizer::new(model);
    let precompiled = Precompiled::from(&spm.precompiled_charsmap).map_err(anyhow::Error::msg)?;
    let replace = Replace::new(ReplacePattern::Regex(" {2,}".to_string()), " ")
        .map_err(anyhow::Error::msg)?;
    tokenizer
        .with_normalizer(Some(Sequence::new(vec![
            precompiled.into(),
            replace.into(),
        ])))
        .map_err(anyhow::Error::msg)?;
    tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, true)));
    tokenizer.with_decoder(Some(MetaspaceDecoder::new(
        '▁',
        PrependScheme::Always,
        true,
    )));
    tokenizer
        .add_special_tokens(
            special_tokens
                .into_iter()
                .map(|(token, _)| AddedToken::from(token, true)),
        )
        .map_err(anyhow::Error::msg)?;
    Ok(tokenizer)
}
