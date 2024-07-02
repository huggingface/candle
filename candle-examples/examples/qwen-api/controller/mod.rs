pub mod pingserver;
// pub mod prompt;
use crate::{config::WebInfo, err::AppError};
use super::SunnyResult;
use askama::Template;
use axum::response::Html;


/// 取得web信息
#[allow(dead_code)]
fn get_web_info<'a>(state: &'a WebInfo) -> WebInfo {
    state.to_owned()
}

/// 取得qwen接口
#[allow(dead_code)]
fn get_textgeneration_info<'a>(state:&'a crate::State) ->crate::State {
    state.to_owned()
}
/// 渲染模板
#[allow(dead_code)]
fn render<T: Template>(tpl: T, handler_name: &str) -> SunnyResult<super::util::types::HtmlResponse> {
    let out = tpl
        .render()
        .map_err(AppError::from)
        .map_err(log_error(handler_name))?;
    Ok(Html(out))
}
/// 记录错误
#[allow(dead_code)]
fn log_error(handler_name: &str) -> Box<dyn Fn(AppError) -> AppError> {
    let handler_name = handler_name.to_string();
    Box::new(move |err| {
        tracing::error!("{}: {:?}", handler_name, err);
        err
    })
}