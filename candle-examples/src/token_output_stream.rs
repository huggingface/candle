use candle::Result;
use candle_transformers::generation::IncrementalDecoder;

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
///
/// It is a thin adapter over [`IncrementalDecoder`], which does the actual work; new code is
/// better off using that directly.
pub struct TokenOutputStream {
    decoder: IncrementalDecoder,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            decoder: IncrementalDecoder::from_tokenizer(tokenizer),
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.decoder.into_tokenizer()
    }

    /// Returns the text that this token made available, if any. `None` means the token did not
    /// settle any new text yet, e.g. it carries the first byte of a multi-byte character.
    ///
    /// The output is append-only: [`IncrementalDecoder::push`] only ever hands out text it has
    /// shown cannot be rewritten, so what this returns is always safe to print or append. For a
    /// tokenizer whose decoder can rewrite text it already produced, that means nothing is
    /// streamed at all and the whole text arrives through [`Self::decode_rest`].
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let text = self.decoder.push(token)?;
        if text.is_empty() {
            Ok(None)
        } else {
            Ok(Some(text.to_string()))
        }
    }

    /// The text held back by [`Self::next_token`], to be printed once generation is over.
    pub fn decode_rest(&self) -> Result<Option<String>> {
        let rest = self.decoder.pending();
        if rest.is_empty() {
            Ok(None)
        } else {
            Ok(Some(rest.to_string()))
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        Ok(self.decoder.text().to_string())
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.decoder
            .tokenizer()
            .get_vocab(true)
            .get(token_s)
            .copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        self.decoder.tokenizer()
    }

    pub fn clear(&mut self) {
        self.decoder.clear()
    }
}
