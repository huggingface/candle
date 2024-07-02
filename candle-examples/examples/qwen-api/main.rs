#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use rayon::Yield;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use async_stream::stream;
use axum::{
    Extension,
    Error,
    response::sse::{Event, Event as SseEvent, KeepAlive,Sse},
};

// use dotenv::dotenv;
use std::{sync::Arc, time::Duration};
use tower_http::trace::TraceLayer;
pub mod config;
pub mod controller;
pub mod err;
pub mod router;
pub mod util;
pub use err::{AppError, AppErrorType};
pub type SunnyResult<T> = std::result::Result<T, crate::AppError>;
use chrono::Local;

use anyhow::{Error as E, Result };

use candle_transformers::models::qwen2::{Config as ConfigBase, Model as ModelBase};
use candle_transformers::models::qwen2_moe::{ Model as ModelMoe};

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use futures::stream::Stream;
use std::convert::Infallible;
// use tokio_stream::StreamExt as _;
// use tokio_stream::wrappers::IntervalStream;

#[derive(Clone)]
pub enum Model {
    Base(ModelBase),
    Moe(ModelMoe),
}

impl Model {
    fn forward(&mut self, xs: &Tensor, s: usize) -> candle::Result<Tensor> {
        match self {
            Self::Moe(ref mut m) => m.forward(xs, s),
            Self::Base(ref mut m) => m.forward(xs, s),
        }
    }
}
#[derive(Clone)]
struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }
    async fn sse_run(&mut self, prompt: String, sample_len: usize) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
        let mut myself=self.clone();
        
        Sse::new(stream! {
            use std::io::Write;
            myself.tokenizer.clear();
            let mut tokens = myself
                .tokenizer
                .tokenizer()
                .encode(prompt.to_owned(), true)
                .map_err(E::msg).unwrap()
                .get_ids()
                .to_vec();
            for &t in tokens.iter() {
                if let Some(t) = myself.tokenizer.next_token(t).unwrap() {
                    yield Ok(
                        SseEvent::default().data(t.clone())
                    );
                    print!("{t}")
                }
            }
            std::io::stdout().flush().unwrap();
            let mut generated_tokens = 0usize;
            let eos_token = match myself.tokenizer.get_token("<|endoftext|>") {
                Some(token) => token,
                None => 0,  // TODO: this is a hack, we should use the tokenizer to get the eos token
            };
            let start_gen = std::time::Instant::now();
            for index in 0..sample_len {
                let context_size = if index > 0 { 1 } else { tokens.len() };
                let start_pos = tokens.len().saturating_sub(context_size);
                let ctxt = &tokens[start_pos..];
                let input = Tensor::new(ctxt, &myself.device).unwrap().unsqueeze(0).unwrap();
                let logits = myself.model.forward(&input, start_pos).unwrap();
                let logits = logits.squeeze(0).unwrap().squeeze(0).unwrap().to_dtype(DType::F32).unwrap();
                let logits = if myself.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = tokens.len().saturating_sub(myself.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        myself.repeat_penalty,
                        &tokens[start_at..],
                    ).unwrap()
                };

                let next_token = myself.logits_processor.sample(&logits).unwrap();
                tokens.push(next_token);
                generated_tokens += 1;
            
                if next_token == eos_token {
                    break;
                }
                if let t = match myself.tokenizer.next_token(next_token) {
                    Ok(t) => t,
                    _ => {
                        if generated_tokens > 1 {
                            break;
                            
                        } else {
                            continue;
                        }
                    }
                } {
                    let t = t;
                    if t==None{
                        continue;
                    }
                    let a =t;
                    let t=a.unwrap();
                    yield Ok(
                        SseEvent::default().data(format!("{t}"))
                    );
                    print!("{t}")
                }
            }
            let dt = start_gen.elapsed();
                if let Some(rest) = myself.tokenizer.decode_rest().map_err(E::msg).unwrap() {
                    yield Ok(
                        SseEvent::default().data(rest.clone())
                    );
                    print!("{rest}");
                }
                std::io::stdout().flush().unwrap();   
            let t=format!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
            yield Ok(
                SseEvent::default().data(t.clone())
            );
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        })
        .keep_alive(KeepAlive::default())
    }
  
    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }


    
}

#[derive(Clone)]
pub struct State {
    pub model: Model,
    pub tokenizer:Tokenizer,
    pub device:Device,
}

// type SharedState = Arc<State>;



#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "0.5b")]
    W0_5b,
    #[value(name = "1.8b")]
    W1_8b,
    #[value(name = "4b")]
    W4b,
    #[value(name = "7b")]
    W7b,
    #[value(name = "14b")]
    W14b,
    #[value(name = "72b")]
    W72b,
    #[value(name = "moe-a2.7b")]
    MoeA27b,
}

#[tokio::main]
async fn main(){
    eprintln!(
            r#"
    ╔══╗
    ╚╗╔╝
    ╔╝(¯`v´¯)
    ╚══`.¸.[ 🇨  ​​🇦  ​​🇳 ​​ 🇩​ ​ ⓛ  🅔​​​ 🌐🌱]​"#);

    

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "candle-examples=debug".into()),  //,tower_http=debug
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    let cfg = config::Config::from_file("./configs/app.toml").unwrap();
    // eprintln!("{:#?}",cfg);
    tracing::info!("读取配置文件成功");
    let local_datetime = Local::now();
    let formatted_datetime = local_datetime.format("%Y-%m-%d %H:%M:%S").to_string();
    let web_info = config::WebInfo {
        web_addr: cfg.web.addr.clone(),
        web_version: "V".to_string()+cfg.web.version.clone().as_str()+"-"+&formatted_datetime,
        info: cfg.info.clone(),
        model: cfg.model.clone(),
    };
    if cfg.web.debug {
        tracing::info!("\n🍤🍎🍤\n基本信息：\n{:#?}\n🍎🍤🍎", web_info );
    }

    // TODO 开始搞模型
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    let api = match Api::new(){
        Ok(api) => api,
        Err(e) => {
            tracing::error!("{}", e);
            return;
        }
    };
    
    let model_id = {
        let size = &cfg.model.model_size;
        format!("Qwen/Qwen1.5-{size}")
    };
    // tracing::error!("model_id:{}",model_id);
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        "main".to_string(),  //args.revision,
    ));

    let tokenizer_filename=match repo.get("tokenizer.json"){
        Ok(tokenizer_filename)=>tokenizer_filename,
        Err(e)=>{
            tracing::error!("{}",e);
            return;
        }
    };
    let filenames=match candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json"){
        Ok(filenames)=>filenames,
        Err(e)=>{
            tracing::error!("{}",e);
            return;
        }
    };
    let tokenizer = match Tokenizer::from_file(tokenizer_filename).map_err(E::msg){
        Ok(tokenizer)=>tokenizer,
        Err(e)=>{
            tracing::error!("{}",e);
            return;
        }
    };
    
    let start = std::time::Instant::now();
    let config_file = match repo.get("config.json"){
        Ok(config_file)=>config_file,
        Err(e)=>{
            tracing::error!("{}",e);
            return;
        }
    };
    // true是使用CPU
    // let _=candle::utils::cuda_is_available();
    let device = match candle_examples::device(false){
        Ok(device)=>device,
        Err(e)=>{
            tracing::error!("{}",e);
            return;
        }
    };

    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    let vb = unsafe { 
        match VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device){
            Ok(vb)=>vb,
            Err(e)=>{
                tracing::error!("{}",e);
                return;
            }
    } };
    let binding = std::fs::read(config_file);
    let reader=match &binding{
        Ok(r)=>r,
        Err(e)=>{
            tracing::error!("{}",e);
            return;
        }
    };
    let config: ConfigBase = match serde_json::from_slice(reader){
        Ok(config)=>config,
        Err(e)=>{
            tracing::error!("{}",e);
            return;
        }
    };
    let model=Model::Base(match ModelBase::new(&config, vb){
        Ok(model)=>model,
        Err(e)=>{
            tracing::error!("{}",e);
            return;
        }
    });
    println!("❤️loaded the model in {:?}❤️", start.elapsed());
    // let start = std::time::Instant::now();
    // let mut pipeline = TextGeneration::new(
    //     model,
    //     tokenizer,
    //     299792458,
    //     Some(1.0),
    //     Some(0.0),
    //     1.1,
    //     64,
    //     &device,
    // );
    // println!("❤️loaded the model in {:?}❤️", start.elapsed());
    let state = State {
        model:model.clone(),
        tokenizer:tokenizer.clone(),
        device:device.clone()
    };
    // let state = Arc::new(State { model:model,tokenizer:tokenizer,device:device });
    // let prompt="C语言写一个冒泡排序算法，并解释其运行原理.";
    // pipeline.run(prompt, 10000).unwrap();
    let app = router::init()
        .layer(TraceLayer::new_for_http())
        .layer(Extension(Arc::new(state.clone())))
        .layer(Extension(Arc::new(web_info.clone())));
    tracing::info!("🌱🌎 服务监听于{}:{}🌐🌱", &cfg.web.addr,&cfg.web.port);
    tokio::join!(
        router::serve(app, cfg.web.port),
    );
}