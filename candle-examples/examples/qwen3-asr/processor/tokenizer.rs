//! Tokenizer wrapper.
//!
//! Qwen3-ASR uses the Qwen2 tokenizer files on HuggingFace. The model repo may
//! ship either:
//! - `tokenizer.json` (preferred, fast tokenizer format), or
//! - `vocab.json` + `merges.txt` + `tokenizer_config.json` (BPE format).
//!
//! For parity with the official stack we must ensure special tokens like
//! `<|im_start|>` and `<|audio_pad|>` are treated as special and encode to a
//! single token id.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};

#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: Option<tokenizers::Tokenizer>,
}

impl Tokenizer {
    pub fn empty() -> Self {
        Self { inner: None }
    }

    /// Load a HuggingFace `tokenizer.json`.
    pub fn from_file(path: &Path) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!(e))
            .with_context(|| format!("failed to load tokenizer from {path:?}"))?;
        Ok(Self {
            inner: Some(tokenizer),
        })
    }

    /// Load tokenizer files from a local model directory.
    pub fn from_pretrained_dir(model_dir: &Path) -> Result<Self> {
        let tokenizer_json = model_dir.join("tokenizer.json");
        if tokenizer_json.exists() {
            return Self::from_file(&tokenizer_json);
        }

        let vocab_json = model_dir.join("vocab.json");
        if !vocab_json.exists() {
            bail!("no tokenizer found in {model_dir:?} (expected tokenizer.json or vocab.json)");
        }

        let merges_txt = model_dir.join("merges.txt");
        let tokenizer_config = model_dir.join("tokenizer_config.json");

        let tokenizer = build_qwen2_bpe_tokenizer(
            &vocab_json,
            merges_txt.exists().then_some(merges_txt.as_path()),
            tokenizer_config
                .exists()
                .then_some(tokenizer_config.as_path()),
        )?;

        Ok(Self {
            inner: Some(tokenizer),
        })
    }

    /// Load tokenizer files from HuggingFace (downloaded into the local cache).
    pub fn from_hf(model_id: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;
        use hf_hub::{Repo, RepoType};

        let api = Api::new().context("failed to create HuggingFace API")?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        // Preferred format: tokenizer.json
        if let Ok(tok) = repo.get("tokenizer.json") {
            let model_dir = tok.parent().map(Path::to_path_buf).ok_or_else(|| {
                anyhow::anyhow!(
                    "failed to determine HuggingFace snapshot directory for {model_id:?}"
                )
            })?;
            return Self::from_pretrained_dir(&model_dir);
        }

        // Fallback format: vocab.json + merges.txt (+ optional tokenizer_config.json)
        let vocab = repo
            .get("vocab.json")
            .with_context(|| format!("failed to download vocab.json for {model_id:?}"))?;
        let _merges = repo
            .get("merges.txt")
            .with_context(|| format!("failed to download merges.txt for {model_id:?}"))?;
        let _tokenizer_config = repo.get("tokenizer_config.json");

        let model_dir = vocab.parent().map(Path::to_path_buf).ok_or_else(|| {
            anyhow::anyhow!("failed to determine HuggingFace snapshot directory for {model_id:?}")
        })?;
        Self::from_pretrained_dir(&model_dir)
    }

    /// Convenience loader: accepts either a local directory path or a HuggingFace model id.
    pub fn from_pretrained(model_id_or_path: &str) -> Result<Self> {
        let path = Path::new(model_id_or_path);
        if path.exists() {
            return Self::from_pretrained_dir(path);
        }
        Self::from_hf(model_id_or_path)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tokenizer = self
            .inner
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tokenizer not loaded"))?;
        let enc = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!(e))
            .context("tokenizer encode failed")?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokenizer = self
            .inner
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tokenizer not loaded"))?;
        tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!(e))
            .context("tokenizer decode failed")
    }

    pub fn token_to_id(&self, token: &str) -> Result<u32> {
        let tokenizer = self
            .inner
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tokenizer not loaded"))?;
        let id = tokenizer
            .token_to_id(token)
            .ok_or_else(|| anyhow::anyhow!("token not found in vocab: {token:?}"))?;
        Ok(id)
    }

    pub fn is_loaded(&self) -> bool {
        self.inner.is_some()
    }

    pub fn require_loaded(&self) -> Result<()> {
        if self.is_loaded() {
            Ok(())
        } else {
            bail!("tokenizer not loaded")
        }
    }
}

fn build_qwen2_bpe_tokenizer(
    vocab_path: &Path,
    merges_path: Option<&Path>,
    config_path: Option<&Path>,
) -> Result<tokenizers::Tokenizer> {
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::NFC;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::pre_tokenizers::sequence::Sequence;
    use tokenizers::pre_tokenizers::split::Split;
    use tokenizers::{SplitDelimiterBehavior, TokenizerImpl};

    // Parse vocab.json -> tokenizers::models::bpe::Vocab, and augment with
    // `added_tokens_decoder` ids so special tokens map to the correct ids.
    let vocab_content = fs::read_to_string(vocab_path)
        .with_context(|| format!("failed to read vocab.json from {vocab_path:?}"))?;
    let mut vocab_std: HashMap<String, u32> = serde_json::from_str(&vocab_content)
        .with_context(|| format!("failed to parse vocab.json from {vocab_path:?}"))?;

    let added_tokens = load_added_tokens_from_config(config_path)?;
    apply_added_tokens_to_vocab(&mut vocab_std, &added_tokens)?;

    let vocab: tokenizers::models::bpe::Vocab = vocab_std.into_iter().collect();

    // Parse merges.txt -> tokenizers::models::bpe::Merges.
    let merges: tokenizers::models::bpe::Merges = if let Some(merges_path) = merges_path {
        let merges_content = fs::read_to_string(merges_path)
            .with_context(|| format!("failed to read merges.txt from {merges_path:?}"))?;
        merges_content
            .lines()
            .skip(1) // Skip header "#version: ..."
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .filter_map(|line| {
                let mut parts = line.split(' ');
                let a = parts.next()?;
                let b = parts.next()?;
                if parts.next().is_some() {
                    return None;
                }
                Some((a.to_string(), b.to_string()))
            })
            .collect()
    } else {
        vec![]
    };

    // Build BPE model.
    let bpe = BPE::new(vocab, merges);

    // GPT-4 style regex pattern (from transformers Qwen2 tokenizer implementation).
    let pretokenize_regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let split = Split::new(pretokenize_regex, SplitDelimiterBehavior::Isolated, false)
        .map_err(|e| anyhow::anyhow!(e))
        .context("failed to build Split pre-tokenizer")?;
    let byte_level = ByteLevel::new(false, true, false);
    let pre_tokenizer = Sequence::new(vec![split.into(), byte_level.into()]);

    // Assemble TokenizerImpl so we can attach normalizer/pre-tokenizer/decoder.
    use tokenizers::{
        DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    };
    type FullTokenizer = TokenizerImpl<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >;
    let mut tokenizer: FullTokenizer = TokenizerImpl::new(bpe);
    tokenizer.with_normalizer(Some(NFC));
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
    tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, true, true)));

    let mut tokenizer: tokenizers::Tokenizer = tokenizer.into();

    // Register added tokens (both special and non-special). This ensures tokens like
    // `<|audio_start|>` are matched as a single token during encoding.
    if !added_tokens.is_empty() {
        let toks: Vec<tokenizers::AddedToken> = added_tokens.into_iter().map(|e| e.token).collect();
        tokenizer.add_tokens(&toks);
    }

    Ok(tokenizer)
}

#[derive(Debug, Clone)]
struct AddedTokenEntry {
    id: u32,
    token: tokenizers::AddedToken,
}

fn load_added_tokens_from_config(config_path: Option<&Path>) -> Result<Vec<AddedTokenEntry>> {
    let Some(config_path) = config_path else {
        return Ok(vec![]);
    };

    let content = fs::read_to_string(config_path)
        .with_context(|| format!("failed to read tokenizer_config.json from {config_path:?}"))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse tokenizer_config.json from {config_path:?}"))?;

    let Some(added_tokens_decoder) = json.get("added_tokens_decoder").and_then(|v| v.as_object())
    else {
        return Ok(vec![]);
    };

    let mut entries: Vec<AddedTokenEntry> = vec![];
    for (id_str, token_info) in added_tokens_decoder {
        let Ok(id) = id_str.parse::<u32>() else {
            continue;
        };
        let Some(content) = token_info.get("content").and_then(|v| v.as_str()) else {
            continue;
        };
        let special = token_info
            .get("special")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let lstrip = token_info
            .get("lstrip")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let rstrip = token_info
            .get("rstrip")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let normalized = token_info
            .get("normalized")
            .and_then(|v| v.as_bool())
            .unwrap_or(!special);
        let single_word = token_info
            .get("single_word")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let token = tokenizers::AddedToken::from(content.to_string(), special)
            .single_word(single_word)
            .lstrip(lstrip)
            .rstrip(rstrip)
            .normalized(normalized);

        entries.push(AddedTokenEntry { id, token });
    }
    entries.sort_by_key(|e| e.id);

    Ok(entries)
}

fn apply_added_tokens_to_vocab(
    vocab: &mut HashMap<String, u32>,
    added_tokens: &[AddedTokenEntry],
) -> Result<()> {
    if added_tokens.is_empty() {
        return Ok(());
    }

    use std::collections::HashSet;
    let mut ids: HashSet<u32> = HashSet::with_capacity(vocab.len());
    for id in vocab.values().copied() {
        if !ids.insert(id) {
            bail!("duplicate id in vocab.json: {id}");
        }
    }

    for entry in added_tokens {
        let tok = entry.token.content.as_str();
        if let Some(existing) = vocab.get(tok).copied() {
            if existing != entry.id {
                bail!(
                    "added token id mismatch for {tok:?}: vocab.json has {existing}, tokenizer_config.json has {}",
                    entry.id
                );
            }
            continue;
        }

        if ids.contains(&entry.id) {
            bail!("added token id collision: id={} token={tok:?}", entry.id);
        }
        vocab.insert(tok.to_string(), entry.id);
        ids.insert(entry.id);
    }

    Ok(())
}
