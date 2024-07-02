use super::{get_web_info,get_textgeneration_info};
use crate::config::WebInfo;
// use super::render;
// use crate::{dbstate::*, util::*};
use axum::{
    //  extract, http::{header::HeaderName, HeaderMap, HeaderValue, StatusCode}, 
     routing::{get,post}, Extension, Router, 
     response::sse::{Event, Sse},
     extract
};
use futures::stream::{self, Stream};
use serde::Deserialize;

use crate::State;

// use serde::{Deserialize,Serialize};
use std::sync::Arc;
use tower_http::trace::TraceLayer;
pub(crate) fn index_router() -> Router {
    Router::new()
        .route("/ping", get(ping))  //获取版本号
        .route("/code", post(code))
        .route("/ssecode", post(ssecode))
        .layer(TraceLayer::new_for_http())
}
#[derive(Debug,Deserialize)]
pub struct PromptPageStruct {
    pub prompt:Option<String>,
}
pub async fn code(
    Extension(state): Extension<Arc<State>>,
    extract::Json(pps): extract::Json<PromptPageStruct>,
)->String{
    let start = std::time::Instant::now();
    let client=get_textgeneration_info(&state);
    let device=&client.device;
    let mut pipeline = crate::TextGeneration::new(
        client.model.clone(),
        client.tokenizer.to_owned(),
        299792458,
        Some(1.0),
        Some(0.0),
        1.1,
        64,
        device,
    );
    pipeline.run(&pps.prompt.unwrap_or("C语言写一个Hello World".to_string()), 10000).unwrap();
    format!("Hello 啊,耗时:{:?}",start.elapsed())
}
pub async fn ssecode(
    Extension(state): Extension<Arc<State>>,
    extract::Json(pps): extract::Json<PromptPageStruct>,
)->Sse<impl stream::Stream<Item = Result<Event, std::convert::Infallible>>>{
    let _start = std::time::Instant::now();
    let client=get_textgeneration_info(&state);
    let device=&client.device;
    let mut pipeline = crate::TextGeneration::new(
        client.model.clone(),
        client.tokenizer.to_owned(),
        299792458,
        Some(1.0),
        Some(0.0),
        1.1,
        64,
        device,
    );
    pipeline.sse_run(pps.prompt.unwrap_or("C语言写一个Hello World".to_string()), 10000).await
}

pub async fn ping(Extension(info): Extension<Arc<WebInfo>>) -> String {
    let info = get_web_info(&info);
    let  budda_bless=r#"
/*
*                             _ooOoo_
*                            o8888888o
*                            88" . "88
*                            (| -_- |)
*                            O\  =  /O
*                         ____/`---'\____
*                       .'  \\|     |//  `.
*                      /  \\|||  :  |||//  \
*                     /  _||||| -:- |||||-  \
*                     |   | \\\  -  /// |   |
*                     | \_|  ''\---/''  |   |
*                     \  .-\__  `-`  ___/-. /
*                   ___`. .'  /--.--\  `. . __
*                ."" '<  `.___\_<|>_/___.'  >'"".
*               | | :  `- \`.;`\ _ /`;.`/ - ` : | |
*               \  \ `-.   \_ __\ /__ _/   .-` /  /
*          ======`-.____`-.___\_____/___.-`____.-'======
*                             `=---='
*          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*                    佛祖保佑       永无BUG
*                    菩提本非树     明镜亦非台
*                    本来无bug      何必常修改
*/
 "#;
    format!("Web version: {}\n{}", info.web_version,budda_bless)
}