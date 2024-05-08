use axum::{
    http::{HeaderMap, StatusCode},
    response::Html,
    Json,
};
pub type Result<T> = std::result::Result<T, crate::err::AppError>;
pub type HandlerResult<T> = self::Result<T>;
pub type RedirectResponse = (StatusCode, HeaderMap, ());
pub type HandlerRedirectResult = self::HandlerResult<RedirectResponse>;
pub type HtmlResponse = Html<String>;
pub type HandlerHtmlResult = HandlerResult<HtmlResponse>;
pub type HandlerJsonResult = (StatusCode, HeaderMap, Json<serde_json::Value>);
