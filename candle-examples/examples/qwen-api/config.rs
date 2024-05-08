use serde::Deserialize;
use tracing_subscriber::field::debug;
#[derive(Deserialize,Debug)]
pub struct WebConfig {
    pub addr: String,
    pub version: String,
    pub debug: bool,
    pub port: u16,
}
// #[derive(Deserialize)]
// pub struct DbConfig {
//     pub pg: String,
//     pub connections: u32,
// }
// #[derive(Deserialize)]
// pub struct MysqlConfig {
//     pub url: String,
//     pub host: String,
//     pub username: String,
//     pub password: String,
//     pub port: u16,
//     pub database: String,
//     pub connections: u32,
// }
// #[derive(Deserialize)]
// pub struct RedisConfig {
//     pub url: String,
// }
// #[derive(Deserialize, Clone, Debug)]
// pub struct InfoPageConfig {
//     pub page_size: u32,
// }
#[derive(Deserialize, Clone, Debug)]
pub struct InfoConfig {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ModelConfig {
    pub model_size: String,
    pub revision: String,
}
// #[derive(Deserialize, Clone, Debug)]
// pub struct LlmOpenai{
//     pub token_url:String,
//     pub url:String,
//     pub openai_url:String,
//     pub key:String,
//     pub secret:String,
//     pub model:String,
// }
// #[derive(Deserialize, Clone, Debug)]
// pub struct LlmConfig {
//     pub openai: LlmOpenai,
// }
#[derive(Deserialize,Debug)]
pub struct Config {
    pub web: WebConfig,
    pub info: InfoConfig,
    pub model: ModelConfig,
}
impl Config {
    /// 从环境读取
    pub fn from_env() -> Result<Self, config::ConfigError> {
        config::Config::builder()
            .add_source(config::Environment::default())
            .build()?
            .try_deserialize()
    }
    /// 从文件读取
    pub fn from_file(path: &'static str) -> Result<Self, config::ConfigError> {
        config::Config::builder()
            .add_source(config::File::with_name(path))
            .add_source(config::Environment::default())
            .build()?
            .try_deserialize()
    }
}
#[derive(Deserialize, Debug, Clone)]
pub struct WebInfo {
    pub web_addr: String,
    pub web_version: String,
    pub model: ModelConfig,
    pub info: InfoConfig,
}
