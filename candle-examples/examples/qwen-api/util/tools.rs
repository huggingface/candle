use chrono::Utc;
use std::error::Error;
/// 取得当前时间
/// return String
pub fn get_now_time() -> String {
    Utc::now().to_rfc3339()
}

/// 字符串转静态字符串
pub fn string_to_static_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}