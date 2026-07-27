//! Incremental detokenization for autoregressive decode loops.
//!
//! An autoregressive loop needs the text produced so far on every step, both to match stop
//! sequences and to stream deltas back to the caller. Re-decoding the whole token sequence on
//! every step is correct but quadratic in the generated length. Appending the decode of each
//! token is linear but wrong: a BPE/SentencePiece decode is *not* the concatenation of its
//! tokens' decodes. A multi-byte character can span several tokens (the prefix decodes to
//! U+FFFD, and a later token retroactively replaces it), and decoding a sub-sequence can gain
//! or lose a leading space.
//!
//! [`IncrementalDecoder`] keeps a bounded trailing window of tokens, re-decodes only that
//! window, and appends the difference against the *previous* window decode, so any artifact
//! caused by decoding a sub-sequence appears in both decodes and cancels out. The window is
//! only moved forward when the shorter decode is provably a suffix of the current one, and
//! only when the tokenizer's decoder has been shown to have a bounded context; otherwise the
//! decoder degrades to whole-sequence decoding rather than corrupting the output.
//!
//! The invariant is that [`IncrementalDecoder::text`] equals a whole-sequence decode of every
//! token pushed so far, at every step.
//!
//! ```no_run
//! use candle_transformers::generation::IncrementalDecoder;
//!
//! # fn main() -> candle::Result<()> {
//! # let tokenizer: tokenizers::Tokenizer = unimplemented!();
//! # let next_token = |_: &[u32]| -> candle::Result<u32> { unimplemented!() };
//! let mut decoder = IncrementalDecoder::new(&tokenizer);
//! let mut tokens = vec![1u32];
//! for _ in 0..128 {
//!     let token = next_token(&tokens)?;
//!     tokens.push(token);
//!     let delta = decoder.push(token)?;
//!     print!("{delta}");
//!     if decoder.text().ends_with("</s>") {
//!         break;
//!     }
//! }
//! print!("{}", decoder.finish());
//! # Ok(())
//! # }
//! ```

use candle::{Error, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

/// The character `String::from_utf8_lossy` (and `ByteFallback`) emits for bytes that do not
/// form a valid UTF-8 sequence yet. A trailing run of these is never emitted, as a later token
/// may complete the sequence and replace them.
const REPLACEMENT: char = '\u{FFFD}';

/// Number of trailing tokens kept in the window after a successful re-anchor.
const MIN_WINDOW_TOKENS: usize = 8;

/// Window length that triggers a re-anchoring attempt. Re-anchoring costs one extra decode of
/// `MIN_WINDOW_TOKENS` tokens and happens at most once every
/// `MAX_WINDOW_TOKENS - MIN_WINDOW_TOKENS` pushes, so the amortized cost per token is constant.
const MAX_WINDOW_TOKENS: usize = 64;

/// Longest pattern rewritten by `tokenizers`' wordpiece `cleanup` helper (`" do not"`).
const CLEANUP_PATTERN_LEN: usize = 7;

/// What could be established about the tokenizer's decoder by inspecting its serialized form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DecoderContext {
    /// Whether the decode of a token depends only on a bounded amount of surrounding context.
    /// When false, windowing is unsound and the whole sequence is re-decoded on every step.
    bounded: bool,
    /// Number of trailing bytes of the decoded text that a future token may still rewrite.
    unstable_tail: usize,
    /// Whether the decode of the last token depends on it being last, so that text only
    /// becomes stable once a further token has been pushed.
    lag_one_step: bool,
}

impl DecoderContext {
    const UNBOUNDED: Self = Self {
        bounded: false,
        unstable_tail: 0,
        lag_one_step: false,
    };

    /// Resolve the bounded-context property of a tokenizer's decoder.
    ///
    /// `tokenizers` keeps the decoder fields private, so the decoder is inspected through its
    /// serialized form. Anything unrecognized is treated as unbounded.
    fn resolve(tokenizer: &Tokenizer) -> Self {
        let decoder = match tokenizer.get_decoder() {
            // Without a decoder `Tokenizer::decode` joins the tokens with a space, which is
            // strictly per-token.
            None => {
                return Self {
                    bounded: true,
                    unstable_tail: 0,
                    lag_one_step: false,
                }
            }
            Some(decoder) => decoder,
        };
        let value = match serde_json::to_value(decoder) {
            Ok(value) => value,
            Err(_) => return Self::UNBOUNDED,
        };
        let mut ctx = Self {
            bounded: true,
            unstable_tail: 0,
            lag_one_step: false,
        };
        // `fused` tracks whether the decoders seen so far have collapsed the token list into a
        // single string. Before that point a decoder only ever sees one token at a time and
        // cannot rewrite across token boundaries; after it, string rewrites apply to the whole
        // decoded text and their reach has to be accounted for.
        let mut fused = false;
        if !scan_decoder(&value, &mut fused, &mut ctx) {
            return Self::UNBOUNDED;
        }
        ctx
    }
}

/// Walk a serialized decoder, accumulating how far a future token can reach back into the text
/// already produced. Returns false as soon as the reach cannot be bounded.
fn scan_decoder(value: &Value, fused: &mut bool, ctx: &mut DecoderContext) -> bool {
    let ty = match value.get("type").and_then(Value::as_str) {
        Some(ty) => ty,
        None => return false,
    };
    match ty {
        "Sequence" => match value.get("decoders").and_then(Value::as_array) {
            Some(decoders) => decoders.iter().all(|d| scan_decoder(d, fused, ctx)),
            None => false,
        },
        // Both collapse the token list into a single string: `Fuse` by joining, `ByteLevel` by
        // concatenating every token's bytes before a single lossy UTF-8 conversion. The latter
        // is what lets a multi-byte character span tokens and be fixed up retroactively.
        "Fuse" | "ByteLevel" => {
            *fused = true;
            true
        }
        // Groups runs of `<0xXX>` tokens before converting them to text, so an incomplete run
        // shows up as a trailing run of U+FFFD which a later token replaces.
        "ByteFallback" => true,
        // Per-token, apart from the prepend scheme which only affects the first token, i.e. the
        // start of the text.
        "Metaspace" => true,
        "WordPiece" => {
            if *fused
                && value
                    .get("cleanup")
                    .and_then(Value::as_bool)
                    .unwrap_or(true)
            {
                ctx.unstable_tail += CLEANUP_PATTERN_LEN;
            }
            true
        }
        "CTC" => {
            if *fused {
                if value
                    .get("cleanup")
                    .and_then(Value::as_bool)
                    .unwrap_or(true)
                {
                    ctx.unstable_tail += CLEANUP_PATTERN_LEN;
                }
                let delimiter = value
                    .get("word_delimiter_token")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                ctx.unstable_tail += delimiter.len();
            }
            true
        }
        // The end-of-word suffix of the *last* token is dropped rather than turned into a
        // space, so the tail of the text changes as soon as another token is pushed.
        "BPEDecoder" => {
            ctx.lag_one_step = true;
            true
        }
        "Strip" => {
            if *fused {
                let stop = value.get("stop").and_then(Value::as_u64).unwrap_or(0) as usize;
                let content = value.get("content").and_then(Value::as_str).unwrap_or("");
                ctx.unstable_tail += stop * content.len();
            }
            true
        }
        "Replace" => {
            if !*fused {
                // Applied to each token independently, so it cannot reach across tokens.
                return true;
            }
            // Applied to the whole text: appending bytes can create a match straddling the
            // boundary, rewriting up to `pattern.len()` bytes already emitted. A regex pattern
            // can match arbitrarily far back, which is exactly the unbounded case.
            match value.get("pattern").and_then(|p| p.get("String")) {
                Some(Value::String(pattern)) => {
                    ctx.unstable_tail += pattern.len();
                    true
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// A detokenizer that turns a stream of token ids into a stream of text deltas in amortized
/// constant time per token.
///
/// See the [module documentation](self) for the approach.
pub struct IncrementalDecoder {
    tokenizer: Tokenizer,
    skip_special_tokens: bool,
    ctx: DecoderContext,
    tokens: Vec<u32>,
    /// Index into `tokens` of the first token of the decode window.
    window_start: usize,
    /// Decode of `tokens[window_start..]`; always a suffix of `text`.
    window_text: String,
    /// Decode of the whole token sequence.
    text: String,
    /// Number of bytes of `text` already returned by `push`.
    emitted: usize,
    /// Bytes of previously returned text invalidated by the last `push`.
    retracted: usize,
}

impl IncrementalDecoder {
    /// Create a decoder for `tokenizer`, resolving the bounded-context property of its decoder.
    ///
    /// The tokenizer is cloned; use [`IncrementalDecoder::from_tokenizer`] to avoid the clone.
    pub fn new(tokenizer: &Tokenizer) -> Self {
        Self::from_tokenizer(tokenizer.clone())
    }

    /// Create a decoder taking ownership of `tokenizer`.
    pub fn from_tokenizer(tokenizer: Tokenizer) -> Self {
        let ctx = DecoderContext::resolve(&tokenizer);
        Self {
            tokenizer,
            skip_special_tokens: true,
            ctx,
            tokens: Vec::new(),
            window_start: 0,
            window_text: String::new(),
            text: String::new(),
            emitted: 0,
            retracted: 0,
        }
    }

    /// Whether special tokens are dropped from the decoded text, `true` by default.
    pub fn with_skip_special_tokens(mut self, skip_special_tokens: bool) -> Self {
        self.skip_special_tokens = skip_special_tokens;
        self.clear();
        self
    }

    /// Whether the tokenizer's decoder was shown to have a bounded context.
    ///
    /// When false the decoder is still correct but re-decodes the whole sequence on every push,
    /// which is quadratic in the number of generated tokens.
    pub fn is_windowed(&self) -> bool {
        self.ctx.bounded
    }

    /// Push a token and return the text that became available because of it.
    ///
    /// The returned slice may be empty: text that a further token could still rewrite - a
    /// trailing run of U+FFFD from an incomplete multi-byte character, or the reach of a
    /// rewriting decoder - is held back until it is stable. Concatenating every value returned
    /// by `push`, followed by [`IncrementalDecoder::finish`], reproduces [`IncrementalDecoder::text`]
    /// exactly, unless [`IncrementalDecoder::retracted`] reports otherwise.
    pub fn push(&mut self, token: u32) -> Result<&str> {
        self.retracted = 0;
        self.tokens.push(token);

        let new_window = self.decode(self.window_start)?;
        let committed = self.text.len() - self.window_text.len();
        // The delta is taken against the *previous* decode of the same window, so a leading
        // space gained or lost by decoding a sub-sequence is present in both and cancels.
        let common = committed + common_prefix_len(&self.window_text, &new_window);
        self.text.truncate(committed);
        self.text.push_str(&new_window);
        self.window_text = new_window;

        if self.ctx.bounded && self.tokens.len() - self.window_start >= MAX_WINDOW_TOKENS {
            self.reanchor()?;
        }

        if common < self.emitted {
            self.retracted = self.emitted - common;
            self.emitted = common;
        }
        let start = self.emitted;
        self.emitted = self.emission_limit(common);
        Ok(&self.text[start..self.emitted])
    }

    /// The whole text decoded so far.
    ///
    /// This is always equal to decoding every pushed token in one go.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The text held back by [`IncrementalDecoder::push`] because a future token could still
    /// rewrite it.
    pub fn pending(&self) -> &str {
        &self.text[self.emitted..]
    }

    /// Release the text held back by [`IncrementalDecoder::push`], for use once generation is
    /// over. Any trailing U+FFFD is genuinely undecodable at that point.
    pub fn finish(&mut self) -> &str {
        let start = self.emitted;
        self.emitted = self.text.len();
        &self.text[start..]
    }

    /// Number of bytes of previously returned text invalidated by the last
    /// [`IncrementalDecoder::push`], to be removed from the end before appending its result.
    ///
    /// This is always zero for a tokenizer whose decoder has a bounded context, which covers
    /// every decoder `tokenizers` ships bar a `Replace` with a regex pattern.
    pub fn retracted(&self) -> usize {
        self.retracted
    }

    /// The tokens pushed so far.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Number of tokens re-decoded by the next [`IncrementalDecoder::push`].
    ///
    /// This stays bounded for a windowed decoder, and grows with the sequence otherwise.
    pub fn window_len(&self) -> usize {
        self.tokens.len() - self.window_start
    }

    /// The underlying tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Consume the decoder, returning the underlying tokenizer.
    pub fn into_tokenizer(self) -> Tokenizer {
        self.tokenizer
    }

    /// Reset the decoder, keeping the tokenizer and its resolved decoder context.
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.window_start = 0;
        self.window_text.clear();
        self.text.clear();
        self.emitted = 0;
        self.retracted = 0;
    }

    fn decode(&self, from: usize) -> Result<String> {
        let tokens = &self.tokens[from..];
        if tokens.is_empty() {
            // Some decoders (`BPEDecoder`) index the last token unconditionally.
            return Ok(String::new());
        }
        self.tokenizer
            .decode(tokens, self.skip_special_tokens)
            .map_err(|err| Error::Msg(format!("cannot decode: {err}")))
    }

    /// Try to move the window forward, keeping the text unchanged.
    ///
    /// The window is only moved when the shorter decode is provably a suffix of the current
    /// window decode, and long enough that a rewriting decoder cannot reach past it. When no
    /// split is acceptable the window simply keeps growing, degrading towards whole-sequence
    /// decoding rather than dropping or duplicating text.
    fn reanchor(&mut self) -> Result<()> {
        let min_tail = self.ctx.unstable_tail.max(1);
        let mut keep = MIN_WINDOW_TOKENS;
        while keep < self.tokens.len() - self.window_start {
            let start = self.tokens.len() - keep;
            let tail = self.decode(start)?;
            if tail.len() >= min_tail && self.window_text.ends_with(&tail) {
                self.window_start = start;
                self.window_text = tail;
                return Ok(());
            }
            keep *= 2;
        }
        Ok(())
    }

    /// Largest prefix of `text` that no future token can rewrite.
    fn emission_limit(&self, common: usize) -> usize {
        let mut limit = self.text.len();
        // An incomplete multi-byte character decodes to U+FFFD, which a later token replaces.
        while limit > 0 && self.text[..limit].ends_with(REPLACEMENT) {
            limit -= REPLACEMENT.len_utf8();
        }
        limit = limit.saturating_sub(self.ctx.unstable_tail);
        while limit > 0 && !self.text.is_char_boundary(limit) {
            limit -= 1;
        }
        if self.ctx.lag_one_step {
            limit = limit.min(common);
        }
        limit.max(self.emitted)
    }
}

/// Length in bytes of the longest common prefix of `a` and `b`, on a character boundary.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut len = 0;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        len += ca.len_utf8();
    }
    len
}
