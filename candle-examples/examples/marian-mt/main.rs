#[cfg(feature = "mkl-unlinked")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::{Parser, ValueEnum};

use candle::{DType, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::marian;

use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    Base,
    Big,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum LanguagePair {
    #[value(name = "fr-en")]
    FrEn,
    #[value(name = "en-zh")]
    EnZh,
    #[value(name = "en-hi")]
    EnHi,
    #[value(name = "en-es")]
    EnEs,
    #[value(name = "en-fr")]
    EnFr,
    #[value(name = "en-ru")]
    EnRu,
}

// TODO: Maybe add support for the conditional prompt.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    tokenizer_dec: Option<String>,

    /// Choose the variant of the model to run.
    #[arg(long, default_value = "big")]
    which: Which,

    // Choose which language pair to use
    #[arg(long, default_value = "fr-en")]
    language_pair: LanguagePair,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use the quantized version of the model.
    #[arg(long)]
    quantized: bool,

    /// Text to be translated
    #[arg(long)]
    text: String,
}

pub fn main() -> anyhow::Result<()> {
    use hf_hub::api::sync::Api;
    let args = Args::parse();

    let config = match (args.which, args.language_pair) {
        (Which::Base, LanguagePair::FrEn) => marian::Config::opus_mt_fr_en(),
        (Which::Big, LanguagePair::FrEn) => marian::Config::opus_mt_tc_big_fr_en(),
        (Which::Base, LanguagePair::EnZh) => marian::Config::opus_mt_en_zh(),
        (Which::Base, LanguagePair::EnHi) => marian::Config::opus_mt_en_hi(),
        (Which::Base, LanguagePair::EnEs) => marian::Config::opus_mt_en_es(),
        (Which::Base, LanguagePair::EnFr) => marian::Config::opus_mt_fr_en(),
        (Which::Base, LanguagePair::EnRu) => marian::Config::opus_mt_en_ru(),
        (Which::Big, lp) => anyhow::bail!("big is not supported for language pair {lp:?}"),
    };
    let tokenizer_default_repo = match args.language_pair {
        LanguagePair::FrEn => "lmz/candle-marian",
        LanguagePair::EnZh
        | LanguagePair::EnHi
        | LanguagePair::EnEs
        | LanguagePair::EnFr
        | LanguagePair::EnRu => "KeighBee/candle-marian",
    };
    let tokenizer = {
        let tokenizer = match args.tokenizer {
            Some(tokenizer) => std::path::PathBuf::from(tokenizer),
            None => {
                let filename = match (args.which, args.language_pair) {
                    (Which::Base, LanguagePair::FrEn) => "tokenizer-marian-base-fr.json",
                    (Which::Big, LanguagePair::FrEn) => "tokenizer-marian-fr.json",
                    (Which::Base, LanguagePair::EnZh) => "tokenizer-marian-base-en-zh-en.json",
                    (Which::Base, LanguagePair::EnHi) => "tokenizer-marian-base-en-hi-en.json",
                    (Which::Base, LanguagePair::EnEs) => "tokenizer-marian-base-en-es-en.json",
                    (Which::Base, LanguagePair::EnFr) => "tokenizer-marian-base-en-fr-en.json",
                    (Which::Base, LanguagePair::EnRu) => "tokenizer-marian-base-en-ru-en.json",
                    (Which::Big, lp) => {
                        anyhow::bail!("big is not supported for language pair {lp:?}")
                    }
                };
                Api::new()?
                    .model(tokenizer_default_repo.to_string())
                    .get(filename)?
            }
        };
        Tokenizer::from_file(&tokenizer).map_err(E::msg)?
    };

    let tokenizer_dec = {
        let tokenizer = match args.tokenizer_dec {
            Some(tokenizer) => std::path::PathBuf::from(tokenizer),
            None => {
                let filename = match (args.which, args.language_pair) {
                    (Which::Base, LanguagePair::FrEn) => "tokenizer-marian-base-en.json",
                    (Which::Big, LanguagePair::FrEn) => "tokenizer-marian-en.json",
                    (Which::Base, LanguagePair::EnZh) => "tokenizer-marian-base-en-zh-zh.json",
                    (Which::Base, LanguagePair::EnHi) => "tokenizer-marian-base-en-hi-hi.json",
                    (Which::Base, LanguagePair::EnEs) => "tokenizer-marian-base-en-es-es.json",
                    (Which::Base, LanguagePair::EnFr) => "tokenizer-marian-base-en-fr-fr.json",
                    (Which::Base, LanguagePair::EnRu) => "tokenizer-marian-base-en-ru-ru.json",
                    (Which::Big, lp) => {
                        anyhow::bail!("big is not supported for language pair {lp:?}")
                    }
                };
                Api::new()?
                    .model(tokenizer_default_repo.to_string())
                    .get(filename)?
            }
        };
        Tokenizer::from_file(&tokenizer).map_err(E::msg)?
    };
    let mut tokenizer_dec = TokenOutputStream::new(tokenizer_dec);

    let device = candle_examples::device(args.cpu)?;
    let vb = {
        let model = match args.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let api = Api::new()?;
                let api = match (args.which, args.language_pair) {
                    (Which::Base, LanguagePair::FrEn) => api.repo(hf_hub::Repo::with_revision(
                        "Helsinki-NLP/opus-mt-fr-en".to_string(),
                        hf_hub::RepoType::Model,
                        "refs/pr/4".to_string(),
                    )),
                    (Which::Big, LanguagePair::FrEn) => {
                        api.model("Helsinki-NLP/opus-mt-tc-big-fr-en".to_string())
                    }
                    (Which::Base, LanguagePair::EnZh) => api.repo(hf_hub::Repo::with_revision(
                        "Helsinki-NLP/opus-mt-en-zh".to_string(),
                        hf_hub::RepoType::Model,
                        "refs/pr/13".to_string(),
                    )),
                    (Which::Base, LanguagePair::EnHi) => api.repo(hf_hub::Repo::with_revision(
                        "Helsinki-NLP/opus-mt-en-hi".to_string(),
                        hf_hub::RepoType::Model,
                        "refs/pr/3".to_string(),
                    )),
                    (Which::Base, LanguagePair::EnEs) => api.repo(hf_hub::Repo::with_revision(
                        "Helsinki-NLP/opus-mt-en-es".to_string(),
                        hf_hub::RepoType::Model,
                        "refs/pr/4".to_string(),
                    )),
                    (Which::Base, LanguagePair::EnFr) => api.repo(hf_hub::Repo::with_revision(
                        "Helsinki-NLP/opus-mt-en-fr".to_string(),
                        hf_hub::RepoType::Model,
                        "refs/pr/9".to_string(),
                    )),
                    (Which::Base, LanguagePair::EnRu) => api.repo(hf_hub::Repo::with_revision(
                        "Helsinki-NLP/opus-mt-en-ru".to_string(),
                        hf_hub::RepoType::Model,
                        "refs/pr/7".to_string(),
                    )),
                    (Which::Big, lp) => {
                        anyhow::bail!("big is not supported for language pair {lp:?}")
                    }
                };
                api.get("model.safetensors")?
            }
        };
        unsafe { VarBuilder::from_mmaped_safetensors(&[&model], DType::F32, &device)? }
    };
    let mut model = marian::MTModel::new(&config, vb)?;

    let mut logits_processor =
        candle_transformers::generation::LogitsProcessor::new(1337, None, None);

    let encoder_xs = {
        let mut tokens = tokenizer
            .encode(args.text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        tokens.push(config.eos_token_id);
        let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
        model.encoder().forward(&tokens, 0)?
    };

    let mut token_ids = vec![config.decoder_start_token_id];
    for index in 0..1000 {
        let context_size = if index >= 1 { 1 } else { token_ids.len() };
        let start_pos = token_ids.len().saturating_sub(context_size);
        let input_ids = Tensor::new(&token_ids[start_pos..], &device)?.unsqueeze(0)?;
        let logits = model.decode(&input_ids, &encoder_xs, start_pos)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;
        let token = logits_processor.sample(&logits)?;
        token_ids.push(token);
        if let Some(t) = tokenizer_dec.next_token(token)? {
            use std::io::Write;
            print!("{t}");
            std::io::stdout().flush()?;
        }
        if token == config.eos_token_id || token == config.forced_eos_token_id {
            break;
        }
    }
    if let Some(rest) = tokenizer_dec.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    println!();
    Ok(())
}
